// server.js
// Zöld Mentor — secure chat backend
// UPDATED: Safer Cloud Run startup + reduced recitation risk for KB answers
// HISTORY RESTORED: Gemini now uses recent chat history again

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

const db = new Firestore({ databaseId: "zoldmentor" });

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

const limiter = rateLimit({
  windowMs: 60 * 1000,
  max: 60,
});

app.use(limiter);

app.get("/health", (_req, res) => {
  res.json({ ok: true });
});

// ─────────────────────────────────────────────────────────────────────────────
// 1) Auth
// ─────────────────────────────────────────────────────────────────────────────
const PUBLIC_API_TOKEN =
  process.env.PUBLIC_API_TOKEN || "zoldmentor-demo-1234567890";

function auth(req, res, next) {
  const authHeader = req.headers.authorization || "";

  const token = authHeader.startsWith("Bearer ")
    ? authHeader.slice(7)
    : req.headers["x-client-token"] || "";

  if (token !== PUBLIC_API_TOKEN) {
    return res.status(401).json({ error: "Unauthorized" });
  }

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
const PROMPT_PATH =
  process.env.PROMPT_PATH || path.join(process.cwd(), "prompts", "base.hu.md");

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
        `[PROMPT] Loaded base.hu.md (${cachedSystemPrompt.length} chars)`
      );
    }
  } catch (e) {
    console.warn(`[PROMPT] Info: ${e.message}`);

    cachedSystemPrompt =
      cachedSystemPrompt ||
      `Te vagy a Zöld Mentor. Magyarul válaszolsz. Rövid, gyakorlatias, oktatási jellegű válaszokat adsz a tudástár alapján.`;
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
  } catch (e) {
    console.error("DB Read Error:", e.message);
    return [];
  }
}

async function saveSession(key, messages) {
  try {
    await db.collection("sessions").doc(key).set({
      messages,
      updatedAt: new Date(),
    });
  } catch (e) {
    console.error("DB Write Error:", e.message);
  }
}

async function loadUserProfile(key) {
  try {
    const doc = await db.collection("profiles").doc(key).get();
    return doc.exists ? doc.data().bio || "" : "";
  } catch (e) {
    console.error("Profile Read Error:", e.message);
    return "";
  }
}

async function saveUserProfile(key, bioText) {
  try {
    await db.collection("profiles").doc(key).set(
      {
        bio: bioText,
        updatedAt: new Date(),
      },
      { merge: true }
    );
  } catch (e) {
    console.error("Profile Write Error:", e.message);
  }
}

// Profile endpoints for the UI
app.get("/get-profile", auth, async (req, res) => {
  try {
    const bio = await loadUserProfile(getConversationKey(req));
    res.json({ ok: true, bio });
  } catch (e) {
    console.error("Get Profile Error:", e.message);
    res.status(500).json({ ok: false });
  }
});

app.post("/update-profile", auth, async (req, res) => {
  try {
    const { bio } = req.body;
    await saveUserProfile(getConversationKey(req), bio || "");
    res.json({ ok: true });
  } catch (e) {
    console.error("Update Profile Error:", e.message);
    res.status(500).json({ ok: false });
  }
});

app.get("/history", auth, async (req, res) => {
  try {
    const hist = await loadSession(getConversationKey(req));

    const messages = hist
      .filter((m) => m && (m.role === "user" || m.role === "assistant"))
      .map((m) => ({
        who: m.role === "user" ? "user" : "bot",
        text: m.content,
      }));

    res.json({ ok: true, messages });
  } catch (e) {
    console.error("History Error:", e.message);
    res.status(500).json({ ok: false });
  }
});

app.post("/log", auth, (req, res) => {
  console.log("📈 Analytics:", JSON.stringify(req.body || {}));
  res.json({ ok: true });
});

// ─────────────────────────────────────────────────────────────────────────────
// 5) KB & Search
// ─────────────────────────────────────────────────────────────────────────────
let kb = { chunks: [], inverted: new Map(), avgdl: 0 };
let retriever = { search: async () => [] };

try {
  kb = loadKB(path.join(process.cwd(), "kb"));

  retriever = createRetriever(kb, {
    geminiApiKey: process.env.GEMINI_API_KEY,
  });

  console.log(`[KB] Loaded ${kb.chunks.length} chunks`);
} catch (e) {
  console.error("[KB] Startup load failed. Continuing without KB search:", e);
}

async function expandQueryWithAI(userQuery) {
  try {
    const fastModel = genAI.getGenerativeModel({
      model: SEARCH_HELPER_MODEL,
    });

    const result = await fastModel.generateContent(`
Identify herbal terms in: "${userQuery}".
Translate to English/Latin scientific names.
Return ONLY keywords spaces.
`);

    return `${userQuery} ${result.response.text().trim()}`;
  } catch (e) {
    console.warn(
      "[SEARCH] Query expansion failed, using original query:",
      e.message
    );
    return userQuery;
  }
}

async function extractKeywordsFromImage(base64Data, mimeType) {
  try {
    const fastModel = genAI.getGenerativeModel({
      model: SEARCH_HELPER_MODEL,
    });

    const result = await fastModel.generateContent([
      {
        text: "Identify the plant or main subject in this image. Return ONLY the Hungarian name, English name, and Latin scientific name.",
      },
      {
        inlineData: {
          mimeType,
          data: base64Data,
        },
      },
    ]);

    return result.response.text().trim();
  } catch (e) {
    console.warn("[IMAGE] Keyword extraction failed:", e.message);
    return "";
  }
}

function safeKbText(text, maxChars = 700) {
  return String(text || "")
    .replace(/\s+/g, " ")
    .slice(0, maxChars);
}

function buildContextBlock(kbHits) {
  if (!kbHits || kbHits.length === 0) {
    return `

Nincs közvetlen találat a tudástárban.
Válaszolj óvatosan, általános ismeretek alapján, és jelezd röviden, hogy a tudástárban nem találtál pontos kapcsolódó részt.`;
  }

  const sources = kbHits
    .slice(0, 3)
    .map(
      (h, i) =>
        `#${i + 1} [${h.source || "tudástár"}]: ${safeKbText(h.text)}`
    )
    .join("\n\n");

  return `

TUDÁSTÁR ADATOK — háttéranyag, nem idézetként használható:
${sources}

Fontos válaszadási szabályok:
- Ne idézz a tudástárból.
- Ne másold a tudástár szavait.
- Ne kövesd a tudástár mondatszerkezetét.
- Ne írj könyvszerű, hosszú bekezdéseket.
- A választ teljesen saját, természetes magyar megfogalmazásban add.
- Röviden, gyakorlatiasan, beszélgetős stílusban válaszolj.
- Ha gyógynövényről van szó, adj használható, óvatos, oktatási jellegű választ.
- Ne adj diagnózist, és ne ígérj gyógyítást.`;
}

// ─────────────────────────────────────────────────────────────────────────────
// 6) Chat Endpoint
// ─────────────────────────────────────────────────────────────────────────────
app.post("/chat", auth, async (req, res) => {
  try {
    const body = req.body || {};

    const userText = body.message || "";
    const imageBase64 = body.image;
    const imageMime = body.mimeType || "image/jpeg";
    const customHistory = body.history || null;

    if (!userText && !imageBase64) {
      return res.status(400).json({ error: "Empty message" });
    }

    const convKey = getConversationKey(req);

    const dbHistory = await loadSession(convKey);
    const activeHistory = customHistory || dbHistory;

    // 1. Build search query
    let searchQuery = userText;

    if (imageBase64) {
      try {
        const imageKeywords = await extractKeywordsFromImage(
          imageBase64,
          imageMime
        );

        searchQuery = `${userText} ${imageKeywords}`.trim();
      } catch (imgErr) {
        console.warn("⚠️ Image keyword extraction bypassed:", imgErr.message);
      }
    }

    // 2. Search KB
    let kbHits = [];

    if (searchQuery.length > 2) {
      try {
        const finalSearchTerm = await expandQueryWithAI(searchQuery);

        kbHits = await retriever.search(finalSearchTerm, { k: 3 });
      } catch (searchError) {
        console.error("⚠️ KB search or query expansion failed:", searchError);

        kbHits = [];
      }
    }

    // 3. Build context
    const contextBlock = buildContextBlock(kbHits);

    // 4. Build instructions
    const basePrompt = buildSystemPrompt();

    const userBio = await loadUserProfile(convKey);

    const bioBlock = userBio
      ? `

USER PROFILE:
${userBio}`
      : "";

    const finalInstruction = `${basePrompt}${bioBlock}${contextBlock}

Válaszadási forma:
- Mindig magyarul válaszolj.
- A válasz legyen természetes, rövid és hasznos.
- Ne használj idézőjeles vagy forrásízű megfogalmazást.
- Ne ismételj hosszú tudástári részeket.
- Ne hivatkozz arra, hogy "a forrás szerint" vagy "a tudástár szerint", hacsak a felhasználó ezt kifejezetten nem kéri.`;

    // 5. Prepare Gemini history
    const rawHistory = activeHistory.slice(-MAX_CONTEXT);
    const recentHistory = [];

    for (const m of rawHistory) {
      const rawText = m.content || m.text || "";
      const sanitizedText = rawText.trim();

      if (!sanitizedText) continue;

      const targetRole = m.role === "assistant" ? "model" : "user";

      if (
        recentHistory.length > 0 &&
        recentHistory[recentHistory.length - 1].role === targetRole
      ) {
        recentHistory[
          recentHistory.length - 1
        ].parts[0].text += `\n${sanitizedText}`;
      } else {
        recentHistory.push({
          role: targetRole,
          parts: [{ text: sanitizedText }],
        });
      }
    }

    while (recentHistory.length > 0 && recentHistory[0].role !== "user") {
      recentHistory.shift();
    }

    const safetySettings = [
      {
        category: "HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold: "BLOCK_NONE",
      },
      {
        category: "HARM_CATEGORY_HARASSMENT",
        threshold: "BLOCK_ONLY_HIGH",
      },
      {
        category: "HARM_CATEGORY_HATE_SPEECH",
        threshold: "BLOCK_ONLY_HIGH",
      },
      {
        category: "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold: "BLOCK_ONLY_HIGH",
      },
    ];

    const generationConfig = {
      temperature: 1.0,
      topP: 0.95,
    };

    const model = genAI.getGenerativeModel({
      model: CHAT_MODEL_NAME,
      systemInstruction: finalInstruction,
      safetySettings,
      generationConfig,
    });

    const chat = model.startChat({
      history: recentHistory,
    });

    // 6. Generate
    let result;

    if (imageBase64) {
      result = await chat.sendMessage([
        {
          text: userText || "Elemzés erről a képről:",
        },
        {
          inlineData: {
            mimeType: imageMime,
            data: imageBase64,
          },
        },
      ]);
    } else {
      result = await chat.sendMessage(userText);
    }

    const reply = result.response.text();

    // 7. Save to DB
    const savedText = imageBase64 ? `[Kép feltöltve] ${userText}` : userText;

    const newHistory = [
      ...dbHistory,
      {
        role: "user",
        content: savedText,
      },
      {
        role: "assistant",
        content: reply,
      },
    ].slice(-MAX_STORAGE);

    await saveSession(convKey, newHistory);

    // Simple profile memory trigger
    const lowerUserText = userText.toLowerCase();

    if (
      lowerUserText.includes("jegyezd meg") ||
      lowerUserText.includes("remember this")
    ) {
      const cleanedFact = userText
        .replace(/remember this|jegyezd meg/gi, "")
        .trim();

      if (cleanedFact) {
        const updatedBio = (userBio ? userBio + "\n" : "") + "- " + cleanedFact;
        await saveUserProfile(convKey, updatedBio);
      }
    }

    return res.json({
      ok: true,
      answer: reply,
    });
  } catch (e) {
    console.error("FULL ERROR DETAIL:", e);

    let userFriendlyError = "Hiba történt a válasz generálásakor.";
    let statusCode = 500;

    const errText = (e.message || "").toLowerCase();

    if (errText.includes("recitation") || errText.includes("filtered")) {
      userFriendlyError =
        "Most nem sikerült biztonságosan megfogalmaznom a választ. Kérlek, próbáld újra egy kicsit konkrétabban, például: „Mire jó a körömvirág külsőleg?” vagy „Hogyan használjam a körömvirágot teaként?”";
      statusCode = 400;
    } else if (
      errText.includes("deadline") ||
      errText.includes("timeout") ||
      errText.includes("exceeded") ||
      e.status === 504
    ) {
      userFriendlyError =
        "A kérés időtúllépés miatt megszakadt a külső szervereken. Kérlek, próbáld meg újra egy pillanat múlva!";
      statusCode = 504;
    } else if (errText.includes("prohibited_content")) {
      userFriendlyError =
        "A választ a biztonsági szűrő blokkolta. Kérlek, fogalmazd meg máshogy a kérdést.";
      statusCode = 400;
    } else if (errText.includes("quota") || e.status === 429) {
      userFriendlyError =
        "Sajnos elértük a mai keretet. Kérlek, próbáld újra később!";
      statusCode = 429;
    } else if (e.status === 413 || errText.includes("too large")) {
      userFriendlyError =
        "A feltöltött kép túl nagy. Kérlek, használj kisebb felbontású fotót.";
      statusCode = 413;
    } else if (errText.includes("safety")) {
      userFriendlyError =
        "A tartalom nem felelt meg a biztonsági irányelveknek. Kérlek, próbáld más szavakkal.";
      statusCode = 400;
    }

    return res.status(statusCode).json({
      error: userFriendlyError,
    });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// 7) Start Server
// ─────────────────────────────────────────────────────────────────────────────
buildSystemPrompt();

const PORT = Number(process.env.PORT) || 8080;

app.listen(PORT, "0.0.0.0", () => {
  console.log(`✅ Zöld Mentor Active on ${PORT}`);
});