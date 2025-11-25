// server.js
// Zöld Mentor — secure chat backend with per-session memory + external prompts + KB (RAG)
// UPDATED: Now powered by Gemini 3 Pro (The "Smartest" Model)

import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import rateLimit from "express-rate-limit";
import { GoogleGenerativeAI } from "@google/generative-ai"; 
import fs from "fs";
import path from "path";
import zlib from "zlib";

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
// 2) AI Clients (Gemini 3 Pro)
// ─────────────────────────────────────────────────────────────────────────────

// Initialize Google Gemini Client
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// We define the model here to check connectivity, but we instantiate per-request in /chat
// Using "gemini-3-pro" for best reasoning and instruction following.
// If you ever need to save money/speed, switch this string back to "gemini-2.5-flash"
const MODEL_NAME = "gemini-3-pro"; 

// ─────────────────────────────────────────────────────────────────────────────
// 3) External prompt loader
// ─────────────────────────────────────────────────────────────────────────────
const PROMPT_PATH =
  process.env.PROMPT_PATH ||
  path.join(process.cwd(), "prompts", "base.hu.md");

let cachedSystemPrompt = null;
let cachedPromptMtime = 0;

function readFileIfExists(p) {
  try {
    return fs.readFileSync(p, "utf8");
  } catch {
    return "";
  }
}

function buildSystemPrompt() {
  try {
    const stat = fs.statSync(PROMPT_PATH);
    if (!cachedSystemPrompt || stat.mtimeMs !== cachedPromptMtime) {
      cachedSystemPrompt = readFileIfExists(PROMPT_PATH);
      cachedPromptMtime = stat.mtimeMs;
      console.log(
        `[PROMPT] Loaded base.hu.md (${PROMPT_PATH}, ${cachedSystemPrompt.length} chars)`
      );
    }
  } catch (e) {
    console.warn(`[PROMPT] Could not read ${PROMPT_PATH}: ${e.message}`);
    cachedSystemPrompt =
      cachedSystemPrompt ||
      "Te vagy a Zöld Mentor. Válaszolj magyarul, világosan.";
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
// 4) Conversation memory (keyed by user ID when available)
// ─────────────────────────────────────────────────────────────────────────────
const SESSIONS = new Map();
const MAX_HISTORY = 12;

function getConversationKey(req) {
  const userId = req.headers["x-user-id"];
  if (userId) return `user:${userId}`;
  const sessionId = req.headers["x-session-id"];
  if (sessionId) return `session:${sessionId}`;
  return `ip:${req.ip || "anon"}`;
}

function getHistory(convKey) {
  if (!SESSIONS.has(convKey)) SESSIONS.set(convKey, []);
  return SESSIONS.get(convKey);
}

function pushToHistory(convKey, msg) {
  const arr = getHistory(convKey);
  arr.push(msg);
  if (arr.length > MAX_HISTORY) arr.splice(0, arr.length - MAX_HISTORY);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4b) Conversation history endpoint
// ─────────────────────────────────────────────────────────────────────────────
app.get("/history", auth, (req, res) => {
  try {
    const convKey = getConversationKey(req);
    const hist = getHistory(convKey) || [];

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

// ─────────────────────────────────────────────────────────────────────────────
// 4c) Analytics endpoint
// ─────────────────────────────────────────────────────────────────────────────
app.post("/log", auth, (req, res) => {
  try {
    const payload = req.body || {};
    console.log("📈 ZM analytics:", JSON.stringify(payload));
  } catch (e) {
    console.warn("⚠️ /log parse error:", e.message);
  }
  res.json({ ok: true });
});

// ─────────────────────────────────────────────────────────────────────────────
// 5) NEW KB SYSTEM — hybrid retriever
// ─────────────────────────────────────────────────────────────────────────────
const kb = loadKB(path.join(process.cwd(), "kb"));

// We use the Gemini API key here because ingest.js used Google embeddings
const retriever = createRetriever(kb, {
  geminiApiKey: process.env.GEMINI_API_KEY,
});

// Quick browser test: /search/debug?q=calendula
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
  res.json({
    ok: true,
    chunks: kb.chunks ? kb.chunks.length : 0,
  });
});

app.get("/system-prompt-preview", auth, (_req, res) => {
  const text = buildSystemPrompt();
  res.json({ ok: true, length: text.length });
});

// ─────────────────────────────────────────────────────────────────────────────
// 6) Chat endpoint (Powered by Gemini 3 Pro)
// ─────────────────────────────────────────────────────────────────────────────
app.post("/chat", auth, async (req, res) => {
  try {
    const body = req.body || {};
    let incoming = Array.isArray(body.messages) ? body.messages : [];
    
    // Fallback for simple request
    if (!incoming.length && body.message) {
      incoming = [{ role: "user", content: String(body.message) }];
    }
    if (!incoming.length)
      return res.status(400).json({ error: "Provide messages or message." });

    // Extract the latest user message
    const lastUser = [...incoming].reverse().find((m) => m.role === "user");
    const userText = lastUser ? String(lastUser.content || "") : "";
    if (!userText)
      return res.status(400).json({ error: "Missing user message." });

    const convKey = getConversationKey(req);
    const history = getHistory(convKey);

    // 1. Retrieve relevant chunks (Search KB)
    const kbHits = await retriever.search(userText, { k: 6 });
    
    // 2. Build System Prompt for Gemini
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

    // 3. Convert History to Gemini Format
    const googleHistory = history.map((m) => {
      return {
        role: m.role === "assistant" ? "model" : "user",
        parts: [{ text: m.content }],
      };
    });

    // 4. Start Gemini 3 Pro Session
    // We use "gemini-3-pro" for maximum reasoning capability
    const model = genAI.getGenerativeModel({ 
      model: MODEL_NAME,
      systemInstruction: finalSystemInstruction 
    });

    const chatSession = model.startChat({
      history: googleHistory,
    });

    // 5. Send Message
    const result = await chatSession.sendMessage(userText);
    const response = await result.response;
    const reply = response.text();

    // 6. Save to local history
    pushToHistory(convKey, { role: "user", content: userText });
    pushToHistory(convKey, { role: "assistant", content: reply });

    res.json({ ok: true, answer: reply });

  } catch (e) {
    console.error("❌ /chat error:", e);
    // Error handling specific to Google API
    if (e.message && e.message.includes("404")) {
       console.error("⚠️ Model not found. Check if 'gemini-3-pro' is available in your region.");
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
  console.log(`📂 KB loaded with ${kb.chunks.length} chunks`);
});