// server.js
// Zöld Mentor — secure chat backend with per-session memory + external prompts + KB (RAG)
// UPDATED: Gemini 3 Pro + Firestore + Extended History for Search

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
// 2) AI Clients (Gemini 3 Pro - Preview)
// ─────────────────────────────────────────────────────────────────────────────
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Use the Preview model (Nov 2025 release)
// Fallback to "gemini-2.5-pro" if preview is unavailable in your region.
const MODEL_NAME = "gemini-3-pro-preview"; 

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
    cachedSystemPrompt = cachedSystemPrompt || "Te vagy a Zöld Mentor. Válaszolj magyarul, világosan.";
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
// 🆕 UPDATED LIMITS:
const MAX_CONTEXT = 12; // Only send last 12 messages to AI (Cheap/Fast)
const MAX_STORAGE = 50; // Keep last 50 messages in DB (For History Search)

function getConversationKey(req) {
  let rawKey = "";
  const userId = req.headers["x-user-id"];
  if (userId) rawKey = `user:${userId}`;
  else {
    const sessionId = req.headers["x-session-id"];
    if (sessionId) rawKey = `session:${sessionId}`;
    else rawKey = `ip:${req.ip || "anon"}`;
  }
  // Sanitize key for Firestore
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
    // Loads the full 50 stored messages so search works on older items
    const hist = await loadSession(convKey);

    const messages = hist
      .filter(
        (m) =>
          m &&
          typeof m.content === "string" &&
          (m.role === "user" || m.role === "assistant")
      )
      .map((m) => ({
        who: m.role === "user" ? "user" : "bot",
        text: m.content,
      }));

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
// 5) KB SYSTEM
// ─────────────────────────────────────────────────────────────────────────────
const kb = loadKB(path.join(process.cwd(), "kb"));
const retriever = createRetriever(kb, { geminiApiKey: process.env.GEMINI_API_KEY });

app.get("/search/debug", async (req, res) => {
  try {
    const q = req.query.q || "calendula";
    const hits = await retriever.search(q, { k: 6 });
    const shaped = hits.map((t) => ({
      source: t.source,
      score: Number(t.score.toFixed(4)),
      preview: t.text.length > 180 ? t.text.slice(0, 180) + "…" : t.text,
    }));
    res.json({ count: shaped.length, results: shaped });
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
// 6) Chat endpoint (Gemini 3 + Firestore Memory)
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

    // 1. Load FULL History from DB (up to 50 messages)
    const convKey = getConversationKey(req);
    const fullHistory = await loadSession(convKey);

    // 2. Search KB
    const kbHits = await retriever.search(userText, { k: 6 });
    
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

    // 4. Convert History for Gemini (Use only last MAX_CONTEXT messages)
    // 🆕 Optimization: We only send the last 12 messages to the Brain to save money.
    const recentHistory = fullHistory.slice(-MAX_CONTEXT);
    
    const googleHistory = recentHistory.map((m) => {
      return {
        role: m.role === "assistant" ? "model" : "user",
        parts: [{ text: m.content }],
      };
    });

    // 5. Generate Answer
    const model = genAI.getGenerativeModel({ 
      model: MODEL_NAME,
      systemInstruction: finalSystemInstruction 
    });

    const chatSession = model.startChat({ history: googleHistory });
    const result = await chatSession.sendMessage(userText);
    const reply = result.response.text();

    // 6. Save NEW History to DB
    // 🆕 We append to fullHistory, then trim to MAX_STORAGE (50)
    const updatedHistory = [
      ...fullHistory, 
      { role: "user", content: userText }, 
      { role: "assistant", content: reply }
    ];

    const historyToSave = updatedHistory.slice(-MAX_STORAGE);

    // Async write (fire and forget) so we don't delay the response
    saveSession(convKey, historyToSave);

    res.json({ ok: true, answer: reply });

  } catch (e) {
    console.error("❌ /chat error:", e);
    if (e.message && e.message.includes("404")) {
       console.error(`⚠️ Model '${MODEL_NAME}' not found. Check availability.`);
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
  console.log(`🧠 AI Brain: ${MODEL_NAME}`);
  console.log(`💾 Memory: Firestore (Context: ${MAX_CONTEXT}, Storage: ${MAX_STORAGE})`);
});