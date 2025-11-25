// server.js
// Zöld Mentor — secure chat backend with per-session memory + external prompts + KB (RAG)
// UPDATED: Gemini 3 Pro + Firestore + AUTOMATIC Query Translation (No manual synonyms needed)

import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import rateLimit from "express-rate-limit";
import { GoogleGenerativeAI } from "@google/generative-ai"; 
import { Firestore } from "@google-cloud/firestore";
import fs from "fs";
import path from "path";

// ⤵️ Imports for the hybrid KB retriever
import { loadKB } from "./lib/kb_loader.js";
import { createRetriever } from "./lib/retriever.js";

// ─────────────────────────────────────────────────────────────────────────────
// 0) Boot
// ─────────────────────────────────────────────────────────────────────────────
dotenv.config();

const app = express();
app.set("trust proxy", 1);
app.use(express.json({ limit: "1mb" }));

// Initialize Firestore (Google's Database)
const db = new Firestore();

// CORS: only allow your sites
const allowedOrigins = [
  "http://localhost:5173",
  "http://localhost:3000",
  "https://academiaeherba.hu",
  "https://www.academiaeherba.hu",
  "https://theherbalconservatory.eu",
  "https://www.theherbalconservatory.eu",
];

app.use(
  cors({
    origin(origin, callback) {
      if (!origin) return callback(null, true);
      if (allowedOrigins.includes(origin)) return callback(null, true);
      return callback(new Error("Not allowed by CORS"));
    },
  })
);

// Rate limit
const limiter = rateLimit({ windowMs: 60 * 1000, max: 120 });
app.use(limiter);

// Health check
app.get("/health", (_req, res) => res.json({ ok: true }));

// ─────────────────────────────────────────────────────────────────────────────
// 1) Auth
// ─────────────────────────────────────────────────────────────────────────────
const PUBLIC_API_TOKEN =
  process.env.PUBLIC_API_TOKEN || "zoldmentor-demo-1234567890";

function auth(req, res, next) {
  const authHeader = req.headers.authorization || "";
  const bearer = authHeader.startsWith("Bearer ") ? authHeader.slice(7) : "";
  const alt = req.headers["x-client-token"] || "";
  const token = bearer || alt;
  const matches = token && token === PUBLIC_API_TOKEN;
  if (!matches) return res.status(401).json({ error: "Unauthorized" });
  return next();
}

// ─────────────────────────────────────────────────────────────────────────────
// 2) AI Clients
// ─────────────────────────────────────────────────────────────────────────────
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// 🧠 MAIN BRAIN: Uses the Smartest Model (Preview) for answering
const CHAT_MODEL_NAME = "gemini-3-pro-preview"; 

// ⚡ TRANSLATOR: Uses the Fastest Model for expanding search terms (Cheap & Fast)
const SEARCH_HELPER_MODEL = "gemini-2.5-flash";

// ─────────────────────────────────────────────────────────────────────────────
// 3) External prompt loader
// ─────────────────────────────────────────────────────────────────────────────
const PROMPT_PATH =
  process.env.PROMPT_PATH ||
  path.join(process.cwd(), "prompts", "base.hu.md");

let cachedSystemPrompt = null;
let cachedPromptMtime = 0;

function readFileIfExists(p) {
  try { return fs.readFileSync(p, "utf8"); } catch { return ""; }
}

function buildSystemPrompt() {
  try {
    const stat = fs.statSync(PROMPT_PATH);
    if (!cachedSystemPrompt || stat.mtimeMs !== cachedPromptMtime) {
      cachedSystemPrompt = readFileIfExists(PROMPT_PATH);
      cachedPromptMtime = stat.mtimeMs;
      console.log(`[PROMPT] Loaded base.hu.md (${PROMPT_PATH}, ${cachedSystemPrompt.length} chars)`);
    }
  } catch (e) {
    console.warn(`[PROMPT] Could not read ${PROMPT_PATH}: ${e.message}`);
    cachedSystemPrompt = cachedSystemPrompt || 
      "Te vagy a Zöld Mentor. Válaszolj magyarul, világosan. Ha összehasonlítást kérnek, használj Markdown táblázatot.";
  }
  return cachedSystemPrompt;
}

app.post("/admin/reload-prompts", auth, (_req, res) => {
  cachedSystemPrompt = null;
  cachedPromptMtime = 0;
  const text = buildSystemPrompt();
  return res.json({ ok: true, length: text.length });
});

// ─────────────────────────────────────────────────────────────────────────────
// 4) PERMANENT DATABASE MEMORY (Firestore)
// ─────────────────────────────────────────────────────────────────────────────
const MAX_CONTEXT = 12; 
const MAX_STORAGE = 50; 

function getConversationKey(req) {
  let rawKey = "";
  const userId = req.headers["x-user-id"];
  if (userId) rawKey = `user:${userId}`;
  else {
    const sessionId = req.headers["x-session-id"];
    if (sessionId) rawKey = `session:${sessionId}`;
    else rawKey = `ip:${req.ip || "anon"}`;
  }
  return rawKey.replace(/\//g, "_");
}

async function loadSession(key) {
  try {
    const doc = await db.collection("sessions").doc(key).get();
    if (!doc.exists) return [];
    return doc.data().messages || [];
  } catch (e) {
    console.error("⚠️ Firestore Read Error:", e.message);
    return [];
  }
}

async function saveSession(key, messages) {
  try {
    await db.collection("sessions").doc(key).set({
      messages: messages,
      updatedAt: new Date()
    });
  } catch (e) {
    console.error("⚠️ Firestore Write Error:", e.message);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4b) Conversation history endpoint
// ─────────────────────────────────────────────────────────────────────────────
app.get("/history", auth, async (req, res) => {
  try {
    const convKey = getConversationKey(req);
    const hist = await loadSession(convKey);
    const messages = hist
      .filter(m => m && (m.role === "user" || m.role === "assistant"))
      .map(m => ({ who: m.role === "user" ? "user" : "bot", text: m.content }));
    res.json({ ok: true, messages });
  } catch (e) {
    console.error("❌ /history error:", e);
    res.status(500).json({ ok: false, error: "History fetch failed" });
  }
});

app.post("/log", auth, (req, res) => {
  try {
    const payload = req.body || {};
    console.log("📈 ZM analytics:", JSON.stringify(payload));
  } catch (e) { console.warn("⚠️ /log parse error:", e.message); }
  res.json({ ok: true });
});

// ─────────────────────────────────────────────────────────────────────────────
// 5) KB SYSTEM & QUERY EXPANDER (The Magic Step)
// ─────────────────────────────────────────────────────────────────────────────
const kb = loadKB(path.join(process.cwd(), "kb"));
const retriever = createRetriever(kb, { geminiApiKey: process.env.GEMINI_API_KEY });

/**
 * 🆕 This function uses a fast AI model to translate Hungarian terms 
 * to English/Latin scientific names BEFORE we search the database.
 */
async function expandQueryWithAI(userQuery) {
  try {
    const fastModel = genAI.getGenerativeModel({ model: SEARCH_HELPER_MODEL });
    
    // We tell the AI strictly to just extract keywords, no chatting.
    const prompt = `
      You are a botanical translator for a search engine. 
      Identify the key herbal/medical terms in this Hungarian query: "${userQuery}".
      Translate them into English and Latin scientific names.
      Return ONLY the translated keywords separated by spaces. No sentences.
    `;

    const result = await fastModel.generateContent(prompt);
    const keywords = result.response.text().trim();
    
    // Combine original query + new English keywords
    // Example: "Mire jó a körömvirág?" + "Calendula officinalis Marigold"
    return `${userQuery} ${keywords}`;
  } catch (e) {
    console.warn("⚠️ Query expansion failed, using original query:", e.message);
    return userQuery;
  }
}

app.get("/search/debug", async (req, res) => {
  try {
    const q = req.query.q || "calendula";
    // Test the expander logic
    const expandedQ = await expandQueryWithAI(q);
    const hits = await retriever.search(expandedQ, { k: 6 });
    
    const shaped = hits.map((t) => ({
      source: t.source,
      score: Number(t.score.toFixed(4)),
      preview: t.text.length > 180 ? t.text.slice(0, 180) + "…" : t.text,
    }));
    
    res.json({ 
      original: q,
      expanded: expandedQ,
      count: shaped.length, 
      results: shaped 
    });
  } catch (e) {
    console.error("❌ /search/debug error:", e.message);
    res.status(500).json({ error: "Search failed" });
  }
});

app.get("/kb-stats", auth, (_req, res) => {
  res.json({ ok: true, chunks: kb.chunks ? kb.chunks.length : 0 });
});

app.get("/system-prompt-preview", auth, (_req, res) => {
  const text = buildSystemPrompt();
  res.json({ ok: true, length: text.length });
});

// ─────────────────────────────────────────────────────────────────────────────
// 6) Chat endpoint
// ─────────────────────────────────────────────────────────────────────────────
app.post("/chat", auth, async (req, res) => {
  try {
    const body = req.body || {};
    let incoming = Array.isArray(body.messages) ? body.messages : [];
    
    if (!incoming.length && body.message) {
      incoming = [{ role: "user", content: String(body.message) }];
    }
    if (!incoming.length) return res.status(400).json({ error: "Provide messages." });

    const lastUser = [...incoming].reverse().find((m) => m.role === "user");
    const userText = lastUser ? String(lastUser.content || "") : "";
    if (!userText) return res.status(400).json({ error: "Missing user message." });

    // 1. Load History
    const convKey = getConversationKey(req);
    const fullHistory = await loadSession(convKey);

    // 2. 🆕 EXPAND QUERY & SEARCH KB
    // This step translates "körömvirág" -> "calendula" silently
    const expandedQuery = await expandQueryWithAI(userText);
    const kbHits = await retriever.search(expandedQuery, { k: 6 });
    
    // 3. Build Prompt
    const baseSystemPromptHu = buildSystemPrompt();
    let contextBlock = "";
    if (kbHits && kbHits.length > 0) {
      const sourcesText = kbHits
        .map((h, i) => `#${i + 1} FORRÁS: ${h.source}\n${h.text}`)
        .join("\n\n---\n\n");
      contextBlock = `\n\nKONTEKSTUS / TUDÁSTÁR:\n${sourcesText}`;
    } else {
      contextBlock = "\n\n(Nincs elérhető speciális tudástár-adat ehhez a kérdéshez.)";
    }

    const finalSystemInstruction = `${baseSystemPromptHu}${contextBlock}`;

    // 4. Convert History
    const recentHistory = fullHistory.slice(-MAX_CONTEXT);
    const googleHistory = recentHistory.map((m) => {
      return {
        role: m.role === "assistant" ? "model" : "user",
        parts: [{ text: m.content }],
      };
    });

    // 5. Generate Answer
    const model = genAI.getGenerativeModel({ 
      model: CHAT_MODEL_NAME,
      systemInstruction: finalSystemInstruction 
    });

    const chatSession = model.startChat({ history: googleHistory });
    const result = await chatSession.sendMessage(userText);
    const reply = result.response.text();

    // 6. Save History
    const updatedHistory = [
      ...fullHistory, 
      { role: "user", content: userText }, 
      { role: "assistant", content: reply }
    ];
    const historyToSave = updatedHistory.slice(-MAX_STORAGE);
    saveSession(convKey, historyToSave);

    res.json({ ok: true, answer: reply });

  } catch (e) {
    console.error("❌ /chat error:", e);
    if (e.message && e.message.includes("404")) {
       console.error(`⚠️ Model '${CHAT_MODEL_NAME}' not found.`);
    }
    res.status(500).json({ error: "Error connecting to AI backend." });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// 8) Start server
// ─────────────────────────────────────────────────────────────────────────────
buildSystemPrompt();

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`✅ Zöld Mentor API listening on port ${PORT}`);
  console.log(`🧠 Main Brain: ${CHAT_MODEL_NAME}`);
  console.log(`⚡ Translator: ${SEARCH_HELPER_MODEL}`);
  console.log(`💾 Memory: Firestore`);
});