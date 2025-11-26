// server.js
// Zöld Mentor — secure chat backend
// UPDATED: Supports Image Uploads (Multimodal)

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
// INCREASED LIMIT: Images are big! We need 10MB limit, not 1MB.
app.use(express.json({ limit: "10mb" })); 

// Connect to 'zoldmentor' database
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

const limiter = rateLimit({ windowMs: 60 * 1000, max: 60 }); // Relaxed for images
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
const CHAT_MODEL_NAME = "gemini-3-pro-preview"; 
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
      `Te vagy a Zöld Mentor. 
      FELADAT: Válaszolj a kérdésekre a megadott tudástár alapján.
      HA KÉPET KAPSZ: Elemzed a képet. Ha növényt látsz, próbáld azonosítani. Ha betegséget látsz, írd le a tüneteket és javasolj gyógynövényes megoldást, de mindig tedd hozzá, hogy "Ez nem orvosi diagnózis".
      Mindig magyarul válaszolj.`;
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
const MAX_CONTEXT = 12; 
const MAX_STORAGE = 50; 

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

app.get("/search/debug", async (req, res) => {
  try {
    const q = req.query.q || "calendula";
    const exp = await expandQueryWithAI(q);
    const hits = await retriever.search(exp, { k: 6 });
    res.json({ original: q, expanded: exp, results: hits.map(t => ({ source: t.source, score: t.score, preview: t.text.slice(0,180) })) });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.get("/kb-stats", auth, (_req, res) => res.json({ chunks: kb.chunks.length }));

// ─────────────────────────────────────────────────────────────────────────────
// 6) Chat Endpoint (With Image Support)
// ─────────────────────────────────────────────────────────────────────────────
app.post("/chat", auth, async (req, res) => {
  try {
    const body = req.body || {};
    const userText = body.message || "";
    const imageBase64 = body.image; // Expecting base64 string (no header)
    const imageMime = body.mimeType || "image/jpeg";

    if (!userText && !imageBase64) return res.status(400).json({ error: "Empty message" });

    const convKey = getConversationKey(req);
    const fullHistory = await loadSession(convKey);

    // 1. Search KB (only if text exists)
    let kbHits = [];
    if (userText.length > 5) {
        const expandedQuery = await expandQueryWithAI(userText);
        kbHits = await retriever.search(expandedQuery, { k: 6 });
    }
    
    // 2. Context
    let contextBlock = "";
    if (kbHits.length > 0) {
      const sources = kbHits.map((h, i) => `#${i+1} [${h.source}]: ${h.text}`).join("\n\n");
      contextBlock = `\n\nTUDÁSTÁR ADATOK (Ezekből dolgozz):\n${sources}`;
    }

    // 3. Instructions
    const basePrompt = buildSystemPrompt();
    const finalInstruction = `${basePrompt}${contextBlock}`;

    // 4. Prepare History for Gemini
    const recentHistory = fullHistory.slice(-MAX_CONTEXT).map(m => ({
      role: m.role === "assistant" ? "model" : "user",
      parts: [{ text: m.content }],
    }));

    const model = genAI.getGenerativeModel({ model: CHAT_MODEL_NAME, systemInstruction: finalInstruction });
    const chat = model.startChat({ history: recentHistory });

    // 5. Send Message (Text + Optional Image)
    let result;
    if (imageBase64) {
        // Multimodal message
        result = await chat.sendMessage([
            { text: userText || "Mit látsz ezen a képen?" },
            { inlineData: { mimeType: imageMime, data: imageBase64 } }
        ]);
    } else {
        // Text only message
        result = await chat.sendMessage(userText);
    }

    const reply = result.response.text();

    // 6. Save to DB (Don't save the huge image string, just a marker)
    const savedText = imageBase64 ? `[Kép feltöltve] ${userText}` : userText;
    
    const newHistory = [...fullHistory, { role: "user", content: savedText }, { role: "assistant", content: reply }].slice(-MAX_STORAGE);
    saveSession(convKey, newHistory);

    res.json({ ok: true, answer: reply });

  } catch (e) {
    console.error("Chat Error:", e);
    res.status(500).json({ error: "Hiba történt a válasz generálásakor." });
  }
});

buildSystemPrompt();
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`✅ Zöld Mentor Active on ${PORT}`));