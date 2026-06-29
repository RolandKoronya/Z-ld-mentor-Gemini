// server.js
// Zöld Mentor — secure chat backend
// UPDATED: Advanced History Alternation Normalization for Gemini 3.1 Strict Validation

import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import rateLimit from "express-rate-limit";
import { GoogleGenerativeAI } from "@google/generative-ai"; 
import { Firestore } from "@google-cloud/firestore";
import fs from "fs";
import path from "path";

import { loadKB } from "./lib/kb_loader.js";
import { createRetriever } from "./lib/retriever.js";

// ─────────────────────────────────────────────────────────────────────────────
// 0) Boot
// ─────────────────────────────────────────────────────────────────────────────
dotenv.config();
const app = express();
app.set("trust proxy", 1);
app.use(express.json({ limit: "10mb" })); 

const db = new Firestore({ databaseId: 'zoldmentor' });

const allowedOrigins = [
  "http://localhost:5173",
  "http://localhost:3000",
  "https://academiaeherba.hu",
  "https://www.academiaeherba.hu",
  "https://theherbalconservatory.eu",
  "https://www.theherbalconservatory.eu",
];

app.use(cors({
  origin(origin, callback) {
    if (!origin) return callback(null, true);
    if (allowedOrigins.includes(origin)) return callback(null, true);
    return callback(new Error("Not allowed by CORS"));
  },
}));

const limiter = rateLimit({ windowMs: 60 * 1000, max: 60 });
app.use(limiter);

app.get("/health", (_req, res) => res.json({ ok: true }));

// ─────────────────────────────────────────────────────────────────────────────
// 1) Auth
// ─────────────────────────────────────────────────────────────────────────────
const PUBLIC_API_TOKEN = process.env.PUBLIC_API_TOKEN || "zoldmentor-demo-1234567890";

function auth(req, res, next) {
  const authHeader = req.headers.authorization || "";
  const token = authHeader.startsWith("Bearer ") ? authHeader.slice(7) : req.headers["x-client-token"] || "";
  if (token !== PUBLIC_API_TOKEN) return res.status(401).json({ error: "Unauthorized" });
  return next();
}

// ─────────────────────────────────────────────────────────────────────────────
// 2) AI Clients
// ─────────────────────────────────────────────────────────────────────────────
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const CHAT_MODEL_NAME = "gemini-3.1-pro-preview";
const SEARCH_HELPER_MODEL = "gemini-2.5-flash";

// ─────────────────────────────────────────────────────────────────────────────
// 3) Prompt Loader
// ─────────────────────────────────────────────────────────────────────────────
const PROMPT_PATH = process.env.PROMPT_PATH || path.join(process.cwd(), "prompts", "base.hu.md");
let cachedSystemPrompt = null;
let cachedPromptMtime = 0;

function readFileIfExists(p) { try { return fs.readFileSync(p, "utf8"); } catch { return ""; } }

function buildSystemPrompt() {
  try {
    const stat = fs.statSync(PROMPT_PATH);
    if (!cachedSystemPrompt || stat.mtimeMs !== cachedPromptMtime) {
      cachedSystemPrompt = readFileIfExists(PROMPT_PATH);
      cachedPromptMtime = stat.mtimeMs;
      console.log(`[PROMPT] Loaded base.hu.md (${cachedSystemPrompt.length} chars)`);
    }
  } catch (e) {
    console.warn(`[PROMPT] Info: ${e.message}`);
    cachedSystemPrompt = cachedSystemPrompt || 
      `Te vagy a Zöld Mentor. FELADAT: Válaszolj a kérdésekre a megadott tudástár alapján. Mindig magyarul válaszolj.`;
  }
  return cachedSystemPrompt;
}

app.post("/admin/reload-prompts", auth, (_req, res) => {
  cachedSystemPrompt = null;
  cachedPromptMtime = 0;
  buildSystemPrompt();
  res.json({ ok: true });
});

// ─────────────────────────────────────────────────────────────────────────────
// 4) Memory & History
// ─────────────────────────────────────────────────────────────────────────────
const MAX_CONTEXT = 24; 
const MAX_STORAGE = 100; 

function getConversationKey(req) {
  const userId = req.headers["x-user-id"];
  if (userId) return `user:${userId}`.replace(/\//g, "_");
  const sessionId = req.headers["x-session-id"];
  if (sessionId) return `session:${sessionId}`.replace(/\//g, "_");
  return `ip:${req.ip || "anon"}`.replace(/\//g, "_");
}

async function loadSession(key) {
  try {
    const doc = await db.collection("sessions").doc(key).get();
    return doc.exists ? doc.data().messages || [] : [];
  } catch (e) { console.error("DB Read Error:", e.message); return []; }
}

async function saveSession(key, messages) {
  try {
    await db.collection("sessions").doc(key).set({ messages, updatedAt: new Date() });
  } catch (e) { console.error("DB Write Error:", e.message); }
}

async function loadUserProfile(key) {
  try {
    const doc = await db.collection("profiles").doc(key).get();
    return doc.exists ? doc.data().bio : "";
  } catch (e) { 
    console.error("Profile Read Error:", e.message); 
    return ""; 
  }
}

async function saveUserProfile(key, bioText) {
  try {
    await db.collection("profiles").doc(key).set({ 
      bio: bioText, 
      updatedAt: new Date() 
    }, { merge: true });
  } catch (e) { 
    console.error("Profile Write Error:", e.message); 
  }
}

// Profile Endpoints for the UI
app.get("/get-profile", auth, async (req, res) => {
  try {
    const bio = await loadUserProfile(getConversationKey(req));
    res.json({ ok: true, bio });
  } catch (e) {
    res.status(500).json({ ok: false });
  }
});

app.post("/update-profile", auth, async (req, res) => {
  try {
    const { bio } = req.body;
    await saveUserProfile(getConversationKey(req), bio);
    res.json({ ok: true });
  } catch (e) {
    res.status(500).json({ ok: false });
  }
});

app.get("/history", auth, async (req, res) => {
  try {
    const hist = await loadSession(getConversationKey(req));
    const messages = hist
      .filter(m => m && (m.role === "user" || m.role === "assistant"))
      .map(m => ({ who: m.role === "user" ? "user" : "bot", text: m.content }));
    res.json({ ok: true, messages });
  } catch (e) { res.status(500).json({ ok: false }); }
});

app.post("/log", auth, (req, res) => {
  console.log("📈 Analytics:", JSON.stringify(req.body || {}));
  res.json({ ok: true });
});

// ─────────────────────────────────────────────────────────────────────────────
// 5) KB & Search
// ─────────────────────────────────────────────────────────────────────────────
const kb = loadKB(path.join(process.cwd(), "kb"));
const retriever = createRetriever(kb, { geminiApiKey: process.env.GEMINI_API_KEY });

async function expandQueryWithAI(userQuery) {
  try {
    const fastModel = genAI.getGenerativeModel({ model: SEARCH_HELPER_MODEL });
    const result = await fastModel.generateContent(`
      Identify herbal terms in: "${userQuery}".
      Translate to English/Latin scientific names.
      Return ONLY keywords spaces.
    `);
    return `${userQuery} ${result.response.text().trim()}`;
  } catch { return userQuery; }
}

async function extractKeywordsFromImage(base64Data, mimeType) {
    try {
        const fastModel = genAI.getGenerativeModel({ model: SEARCH_HELPER_MODEL });
        const result = await fastModel.generateContent([
            { text: "Identify the plant or main subject in this image. Return ONLY the Hungarian name, English name, and Latin scientific name." },
            { inlineData: { mimeType: mimeType, data: base64Data } }
        ]);
        return result.response.text().trim();
    } catch (e) { return ""; }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6) Chat Endpoint (Context Aware & Error Specific)
// ─────────────────────────────────────────────────────────────────────────────
app.post("/chat", auth, async (req, res) => {
  try {
    const body = req.body || {};
    const userText = body.message || "";
    const imageBase64 = body.image; 
    const imageMime = body.mimeType || "image/jpeg";
    const customHistory = body.history || null;

    if (!userText && !imageBase64) return res.status(400).json({ error: "Empty message" });

    const convKey = getConversationKey(req);
    const dbHistory = await loadSession(convKey);
    let activeHistory = customHistory || dbHistory;

    // 1. Build Search Query (Wrapped in safe fallback)
    let searchQuery = userText;
    if (imageBase64) {
        try {
            const imageKeywords = await extractKeywordsFromImage(imageBase64, imageMime);
            searchQuery = `${userText} ${imageKeywords}`;
        } catch (imgErr) {
            console.warn("⚠️ Image keyword extraction lagged out, bypassing...", imgErr);
        }
    }
    
    // 2. Search KB (Protected with a try/catch to stop execution timeouts)
    let kbHits = [];
    if (searchQuery.length > 2) {
        try {
            const finalSearchTerm = await expandQueryWithAI(searchQuery);
            kbHits = await retriever.search(finalSearchTerm, { k: 6 });
        } catch (searchError) {
            console.error("⚠️ KB search or query expansion stalled, utilizing fallback knowledge:", searchError);
            kbHits = []; 
        }
    }
    
    // 3. Context
    let contextBlock = "";
    if (kbHits.length > 0) {
      const sources = kbHits.map((h, i) => `#${i+1} [${h.source}]: ${h.text}`).join("\n\n");
      contextBlock = `\n\nTUDÁSTÁR ADATOK (Ezekből dolgozz):\n${sources}`;
    } else {
      contextBlock = "\n\n(Nincs találat a tudástárban. Használd az általános tudásodat, de jelezd.)";
    }

    // 4. Build Instructions
    const basePrompt = buildSystemPrompt();

    const userBio = await loadUserProfile(convKey);
    const bioBlock = userBio ? `\n\nUSER PROFILE (Always remember these facts about the user):\n${userBio}` : "";

    const finalInstruction = `${basePrompt}${bioBlock}${contextBlock}`;

    // 🆕 5. Prepare Gemini History (Strict Structural Schema Builder)
    let rawHistory = activeHistory.slice(-MAX_CONTEXT);
    let recentHistory = [];

    for (const m of rawHistory) {
      const rawText = m.content || m.text || "";
      const sanitizedText = rawText.trim();
      if (!sanitizedText) continue; // Drop completely blank messages

      const targetRole = m.role === "assistant" ? "model" : "user";

      // If the last added message has the SAME role, merge texts to protect strict alternation rule
      if (recentHistory.length > 0 && recentHistory[recentHistory.length - 1].role === targetRole) {
        recentHistory[recentHistory.length - 1].parts[0].text += `\n${sanitizedText}`;
      } else {
        recentHistory.push({
          role: targetRole,
          parts: [{ text: sanitizedText }],
        });
      }
    }

    // Make sure the history list strictly initiates with a 'user' turn
    while (recentHistory.length > 0 && recentHistory[0].role !== "user") {
      recentHistory.shift();
    }

    const safetySettings = [
      { category: "HARM_CATEGORY_DANGEROUS_CONTENT", threshold: "BLOCK_NONE" },
      { category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_ONLY_HIGH" },
      { category: "HARM_CATEGORY_HATE_SPEECH", threshold: "BLOCK_ONLY_HIGH" },
      { category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold: "BLOCK_ONLY_HIGH" },
    ];

    const model = genAI.getGenerativeModel({ 
        model: CHAT_MODEL_NAME, 
        systemInstruction: finalInstruction,
        safetySettings 
    });
    
    const chat = model.startChat({ history: recentHistory });

    // 6. Generate
    let result;
    if (imageBase64) {
        result = await chat.sendMessage([
            { text: userText || "Elemzés erről a képről:" },
            { inlineData: { mimeType: imageMime, data: imageBase64 } }
        ]);
    } else {
        result = await chat.sendMessage(userText);
    }

    const reply = result.response.text();

    // 7. Save to DB
    const savedText = imageBase64 ? `[Kép feltöltve] ${userText}` : userText;
    const newHistory = [...dbHistory, { role: "user", content: savedText }, { role: "assistant", content: reply }].slice(-MAX_STORAGE);
    saveSession(convKey, newHistory);

    if (userText.toLowerCase().includes("jegyezd meg") || userText.toLowerCase().includes("remember this")) {
        const cleanedFact = userText.replace(/remember this|jegyezd meg/gi, "").trim();
        const updatedBio = (userBio ? userBio + "\n" : "") + "- " + cleanedFact;
        await saveUserProfile(convKey, updatedBio);
    }

    res.json({ ok: true, answer: reply });

  } catch (e) {
    console.error("FULL ERROR DETAIL:", e);

    let userFriendlyError = "Hiba történt a válasz generálásakor.";
    const errText = (e.message || "").toLowerCase();

    if (errText.includes("deadline") || errText.includes("timeout") || errText.includes("exceeded") || e.status === 504) {
      userFriendlyError = "A kérés időtúllépés miatt megszakadt a külső szervereken. Kérlek, próbáld meg újra egy pillanat múlva!";
    } else if (errText.includes("prohibited_content")) {
      userFriendlyError = "A választ a biztonsági szűrő blokkolta. Kérlek, fogalmazd meg máshogy a kérdést (kerüld az orvosi diagnózis jellegű kéréseket).";
    } else if (errText.includes("quota") || e.status === 429) {
      userFriendlyError = "Sajnos elértük a mai ingyenes keretünket. Kérlek, próbáld újra később!";
    } else if (e.status === 413 || errText.includes("too large")) {
      userFriendlyError = "A feltöltött kép túl nagy. Kérlek, használj kisebb felbontású fotót (max 10MB).";
    } else if (errText.includes("safety")) {
      userFriendlyError = "A tartalom nem felelt meg a biztonsági irányelveknek. Kérlek, próbáld más szavakkal.";
    }

    res.status(500).json({ error: userFriendlyError });
  }
});

buildSystemPrompt();
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`✅ Zöld Mentor Active on ${PORT}`));