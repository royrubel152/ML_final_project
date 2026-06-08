# RAG Architecture — Deep Explanation
## MBA Academic Advisor Chatbot — Hebrew University Business School

**Document type:** Technical + conceptual explanation  
**Audience:** Developers, product managers, academic stakeholders  
**Date:** May 2026

---

## Table of Contents

1. [The Core Problem — Why Not Just Ask Gemini?](#1-the-core-problem)
2. [Solution 1 — Context Stuffing (Our First Version)](#2-solution-1--context-stuffing)
3. [Solution 2 — RAG (Our Current Version)](#3-solution-2--rag)
4. [How RAG Works — Naked Step-by-Step Process](#4-how-rag-works--naked-step-by-step)
5. [The Magic of Vector Embeddings](#5-the-magic-of-vector-embeddings)
6. [FAISS — How Similarity Search Works](#6-faiss--how-similarity-search-works)
7. [Why RAG is Better — Concrete Comparison](#7-why-rag-is-better--concrete-comparison)
8. [Problems RAG Solves](#8-problems-rag-solves)
9. [What Happens in Our System Right Now](#9-what-happens-in-our-system-right-now)
10. [Remaining Limitations](#10-remaining-limitations)

---

## 1. The Core Problem

### What is an LLM?

A Large Language Model (LLM) like Google Gemini is a system trained on hundreds of billions of text documents from the internet — Wikipedia, books, websites, forums, academic papers. During training it learns statistical patterns in language and develops a broad understanding of the world.

**The key word is: trained.**

Training happens once, before the model is deployed. After training, the model's knowledge is frozen. It knows what was in the internet up to its training cutoff date. It does not know:

- What changed after training
- Private documents that were never on the internet
- Internal institutional rules, regulations, or policies
- A specific university's specific MBA program rules

### What happens when you ask Gemini without giving it context?

```
Student: "מה הממוצע הדרוש לקבלה לתוכנית MBA של האוניברסיטה העברית?"

Gemini thinks:
  - I was trained on general text about MBA programs
  - I know that most MBA programs require certain GPA thresholds
  - Israeli universities typically require around 80-85 average
  - I'll generate a confident-sounding answer based on this
  
Gemini answers: "בדרך כלל נדרש ממוצע של 80 ומעלה..."
```

This answer might be **plausible but wrong**. Gemini is pattern-matching from general knowledge, not reading the actual HUJI regulations. This is called **hallucination** — the model generates text that sounds correct but is not grounded in fact.

### Why is hallucination dangerous here?

A student relying on a hallucinated answer might:
- Submit an application they don't qualify for
- Miss a real exemption they're entitled to
- Make wrong course selection decisions
- Misunderstand graduation requirements

In an academic context, wrong information has real consequences.

---

## 2. Solution 1 — Context Stuffing (Our First Version)

### The idea

Instead of letting Gemini answer from memory, we provide the actual HUJI documents inside the message. Gemini reads what we give it and answers from that.

```
We scraped 6 official HUJI pages + 1 PDF
Total: ~38,000 characters of official content

Every message to Gemini looked like this:

┌─────────────────────────────────────────────────────────┐
│ SYSTEM PROMPT:                                          │
│   You are an MBA advisor. Answer ONLY from the          │
│   official content below.                              │
│                                                         │
│ OFFICIAL CONTENT (38,000 chars):                        │
│   === תקנון אקדמי ===                                   │
│   [full PDF content here]                               │
│   === פטורים ===                                        │
│   [full exemptions page content]                        │
│   === קבלה ===                                          │
│   [full admissions page content]                        │
│   ... (all 6 sources)                                   │
│                                                         │
│ USER MESSAGE:                                           │
│   מה הממוצע הדרוש לקבלה?                               │
└─────────────────────────────────────────────────────────┘
```

### Why this was better than asking Gemini directly

- Gemini reads the actual HUJI rules — not its training data
- Answers are grounded in real official content
- Simple to implement — no extra infrastructure needed

### Why this was still not good enough

**Problem 1 — Wasted tokens**

Every single request sent all 38,000 chars to Gemini, even for a simple question. 38,000 chars ≈ 10,000 tokens. We were paying (in compute and money) for Gemini to read 50 pages every time a student asked one question.

**Problem 2 — Needle in a haystack**

When Gemini receives 10,000 tokens, it must process all of it to find the relevant part. Research shows LLMs perform worse when the relevant information is buried deep in a long context — they tend to focus on the beginning and end of the prompt and miss content in the middle.

**Problem 3 — Does not scale**

38,000 chars fits in Gemini's context window today. But what happens when the university wants to add:
- The full course catalog (200+ courses)
- Registration FAQ
- Past student questions database
- Student handbook

Suddenly you have 500,000 chars. That is too large for the context window, and even if it fit, the quality would degrade severely.

**Problem 4 — No precision**

Every answer draws from all sources equally. A question about exemptions triggers the bot to also "read" the admissions page, the accelerated track page, and the full PDF — none of which are relevant. This increases hallucination risk because Gemini has more irrelevant content to accidentally pull from.

---

## 3. Solution 2 — RAG (Our Current Version)

### What is RAG?

**RAG = Retrieval-Augmented Generation**

Instead of sending everything to Gemini every time, we:
1. **Retrieve** only the relevant pieces of content for this specific question
2. **Augment** the question with those pieces
3. **Generate** an answer using only the relevant content

The key innovation: we only send Gemini what it needs to answer this specific question.

```
Question: "מה הממוצע הדרוש לקבלה?"

OLD WAY:  Send 38,000 chars → Gemini reads everything → answers
NEW WAY:  Find top 5 relevant chunks → Send ~1,500 chars → Gemini answers
```

Same quality of answer. 85% less tokens. Higher precision.

---

## 4. How RAG Works — Naked Step-by-Step Process

### Phase 1 — Indexing (happens once at startup)

This is the preparation phase. It happens when the server starts and only repeats if the content changes.

```
STEP 1: SCRAPE
─────────────
Playwright opens 6 HUJI pages + downloads PDF
Extracts full rendered text
Result: 38,000 chars of official content

STEP 2: CHUNK
─────────────
Split the 38,000 chars into 63 overlapping pieces
Each piece = ~800 characters
Overlap = 150 characters (so context doesn't get cut mid-sentence)

Example chunks from the admissions page:
  Chunk 8:  "...תנאי הקבלה לתוכנית MBA: ממוצע ציונים מצטבר של 85..."
  Chunk 9:  "...85 ומעלה בתואר הראשון. פטור מלימודי אנגלית. ציון GMAT..."
  Chunk 10: "...GMAT בציון 40 ומעלה בחלק הכמותי, או ציון 130 בסכום..."

Note: chunks overlap — chunk 9 starts 150 chars before chunk 8 ends.
This ensures no information is lost at chunk boundaries.

STEP 3: EMBED
─────────────
Each chunk is passed to the embedding model (gemini-embedding-001)
The model converts the text into a vector — a list of 3,072 numbers
that mathematically represents the MEANING of that text

Chunk 8  → [0.023, -0.891, 0.445, 0.112, -0.334, ... ] (3,072 numbers)
Chunk 9  → [0.019, -0.876, 0.431, 0.098, -0.321, ... ] (3,072 numbers)
Chunk 10 → [0.021, -0.869, 0.428, 0.103, -0.318, ... ] (3,072 numbers)

These three chunks are about admission requirements → their vectors are CLOSE.

A chunk about exemptions would have very different numbers → far away.

STEP 4: BUILD FAISS INDEX
─────────────────────────
All 63 vectors (63 × 3,072 numbers) are stored in a FAISS index
FAISS = Facebook AI Similarity Search
It is a database optimized to find "nearest neighbor" vectors instantly

The index is saved to disk: data/faiss_index.bin
The chunks are saved to disk: data/chunks.json

Next server restart: load from disk in 1 second (no re-embedding needed)
```

### Phase 2 — Query (happens for every student message)

```
STEP 1: STUDENT ASKS A QUESTION
───────────────────────────────
"אני בשנה ב' בכלכלה, ממוצע 86. האם אני יכול להגיש לתוכנית?"

STEP 2: EMBED THE QUESTION
──────────────────────────
The same embedding model converts the question to a vector:
Query → [0.018, -0.881, 0.429, 0.095, -0.315, ... ] (3,072 numbers)

STEP 3: SIMILARITY SEARCH IN FAISS
───────────────────────────────────
FAISS compares the query vector to all 63 chunk vectors
Finds the 5 closest chunks by cosine similarity

Result:
  Chunk 8  — score 0.91 — "ממוצע ציונים מצטבר של 85 ומעלה..."
  Chunk 9  — score 0.87 — "פטור מלימודי אנגלית. ציון GMAT..."
  Chunk 31 — score 0.82 — "תנאי קבלה למסלול המואץ לסטודנטים..."
  Chunk 12 — score 0.79 — "בעלי תואר ראשון בכלכלה בציון 90..."
  Chunk 14 — score 0.74 — "ניתן להגיש בקשה לשקילת מועמדות ללא GMAT..."

STEP 4: BUILD AUGMENTED PROMPT
───────────────────────────────
Combine the 5 chunks + original question:

"ענה על השאלה הבאה בהתבסס ONLY על המקטעים הבאים:

[מקור: קבלה | bschool.huji.ac.il/mba/admittance]
ממוצע ציונים מצטבר של 85 ומעלה בתואר הראשון...

[מקור: קבלה | bschool.huji.ac.il/mba/admittance]
פטור מלימודי אנגלית. ציון GMAT...

... (3 more chunks)

שאלה: אני בשנה ב' בכלכלה, ממוצע 86. האם אני יכול להגיש לתוכנית?"

Total: ~1,500 tokens (vs 10,000 before)

STEP 5: GEMINI ANSWERS
──────────────────────
Gemini reads ONLY the 5 relevant chunks
Generates a structured Hebrew answer
Based exclusively on the retrieved content

STEP 6: CITATION VERIFICATION
──────────────────────────────
Our code checks: does every quoted sentence actually appear in the chunks?
If not → replace with "[ציטוט לא אומת — פנה למזכירות]"

STEP 7: SAVE SESSION
─────────────────────
Conversation history saved to data/sessions/<session_id>.json
Next message by this student will load history → follow-up questions work
```

---

## 5. The Magic of Vector Embeddings

This is the most important concept to understand.

### What is an embedding?

An embedding model is a neural network that was trained to convert text into numbers such that **similar meanings produce similar numbers**.

Think of it like a coordinate system for meaning:
- Text about admissions → coordinates near (0.02, -0.88, 0.43, ...)
- Text about exemptions → coordinates near (0.31, 0.12, -0.67, ...)
- Text about courses → coordinates near (-0.45, 0.67, 0.21, ...)

### Why this is powerful

A student asks: **"כמה צריך ממוצע?"**
The document says: **"ממוצע ציונים מצטבר של 85 ומעלה בתואר הראשון"**

These sentences use different words but mean the same thing. A keyword search would fail. An embedding search succeeds because both are mathematically close in meaning-space.

### Hebrew works because of multilingual training

The embedding model (gemini-embedding-001) was trained on Hebrew text. It understands:
- "ממוצע" and "ציון ממוצע" are the same concept
- "קבלה" and "תנאי קבלה" are related
- "מסלול מואץ" and "fast track" map to nearby coordinates

This is why RAG works for Hebrew academic text without translation.

---

## 6. FAISS — How Similarity Search Works

### The math behind it

When the student asks a question, we embed it to get a query vector Q.
We have 63 chunk vectors: C1, C2, ..., C63.

We want to find the chunks whose vectors are most similar to Q.

**Cosine similarity** measures the angle between two vectors:
- Score = 1.0 → identical meaning
- Score = 0.9 → very similar
- Score = 0.7 → somewhat related
- Score = 0.4 → barely related
- Score = 0.0 → completely unrelated

We retrieve only chunks with score above 0.45 (our minimum threshold).

### Why FAISS and not a simple loop?

With 63 chunks, a loop is fine. But FAISS was designed for millions of vectors. It uses optimized C++ code and special data structures (inverted file index, HNSW graphs) to find nearest neighbors in microseconds even with 100 million vectors.

We use `IndexFlatIP` (inner product = cosine similarity after L2 normalization). This is the most accurate FAISS index — exhaustive search, no approximation. Correct for our scale.

---

## 7. Why RAG is Better — Concrete Comparison

```
SCENARIO: Student asks "מה תנאי הקבלה?"
```

### Gemini alone (no context)

```
Input:  50 tokens (question only)
Source: Gemini's training data (frozen 2024)
Output: Generic MBA admission answer
Risk:   HIGH hallucination — invents HUJI-specific rules
Cost:   Lowest tokens, but WRONG
```

### Context stuffing (our first version)

```
Input:  10,000 tokens (38K chars of all HUJI content + question)
Source: All 6 HUJI pages injected every time
Output: Usually correct, but Gemini reads 50 pages for 1 question
Risk:   MEDIUM hallucination — grounded but Gemini can still mix content
Cost:   10x more tokens per request
```

### RAG (current version)

```
Input:  ~1,500 tokens (5 relevant chunks + question)
Source: Only the 5 chunks most relevant to THIS question
Output: Precise answer from exactly the right sections
Risk:   LOW hallucination — Gemini only sees relevant content
Cost:   85% fewer tokens than context stuffing
```

### Full comparison table

| Dimension | Gemini alone | Context stuffing | RAG |
|---|---|---|---|
| Answer accuracy | Low | Medium | High |
| Hallucination risk | Very high | Medium | Low |
| Tokens per request | ~50 | ~10,000 | ~1,500 |
| Cost per 1000 queries | $0.01 | $1.50 | $0.23 |
| Scales to large docs | Yes (wrong) | No | Yes |
| Citation grounding | None | Weak | Strong |
| Startup time | Instant | 30 seconds | 90s first time, 5s after |
| Answer relevance | Generic | Broad | Precise |

---

## 8. Problems RAG Solves

### Problem 1 — Hallucination
**Before:** Gemini invents HUJI-specific rules from general knowledge  
**After:** Gemini only sees verified official chunks — can't invent what isn't there

### Problem 2 — Wasted tokens and cost
**Before:** 10,000 tokens per request (all content every time)  
**After:** ~1,500 tokens per request (only relevant chunks)  
**Savings:** 85% reduction in API cost

### Problem 3 — Scalability
**Before:** Adding more sources would overflow the context window  
**After:** Can index millions of documents — only top 5 chunks ever sent to Gemini

### Problem 4 — Precision
**Before:** Gemini reads all content including unrelated sections  
**After:** Only sections semantically relevant to the question are retrieved

### Problem 5 — Stale sessions
**Before:** All conversations lost on server restart  
**After:** Sessions saved to disk — students can continue conversations after restart

### Problem 6 — Citation accuracy
**Before:** Gemini quoted invented text (hallucinated page numbers and sentences)  
**After:** Citations verified against retrieved chunks programmatically — fake quotes replaced automatically

---

## 9. What Happens in Our System Right Now

### Full architecture diagram

```
═══════════════════════════════════════════════════════════════
                     STARTUP PHASE (once)
═══════════════════════════════════════════════════════════════

scrape_runner.py (subprocess)
  ├── Playwright opens 6 HUJI URLs
  ├── Waits for JS to render
  ├── Extracts full body text
  └── Downloads + parses academic PDF
            │
            ▼ 38,000 chars
scraper.py
  ├── Saves to data/scraped_content.json (cache)
  └── Returns content string
            │
            ▼
rag.py — get_or_build_index()
  ├── Checks content hash vs saved hash
  ├── If same → loads faiss_index.bin (1 second)
  └── If changed →
        ├── Chunks content into 63 pieces (800 chars, 150 overlap)
        ├── Embeds each chunk via gemini-embedding-001
        ├── Builds FAISS IndexFlatIP (3,072 dim)
        └── Saves index + chunks + hash to disk
            │
            ▼
app.py — lifespan()
  ├── Loads FAISS index into memory
  ├── Initializes Gemini 2.5 Flash with rules-only system prompt
  ├── Cleans up expired sessions
  └── Server ready


═══════════════════════════════════════════════════════════════
                   PER REQUEST (every question)
═══════════════════════════════════════════════════════════════

Student browser (script.js)
  └── POST /chat { message, session_id }
            │
            ▼
app.py — /chat endpoint
  │
  ├── 1. RETRIEVE
  │     rag.retrieve(user_message, faiss_index, chunks)
  │       ├── Embed question → 3,072-dim vector
  │       ├── FAISS search → top 5 by cosine similarity
  │       └── Filter: score > 0.45
  │
  ├── 2. AUGMENT
  │     rag.build_rag_prompt(question, top_5_chunks)
  │       └── Builds: [source1 chunk] + [source2 chunk] + ... + question
  │
  ├── 3. LOAD SESSION
  │     sessions.load_history(session_id)
  │       └── Reads data/sessions/<id>.json → chat history
  │
  ├── 4. GENERATE
  │     gemini.start_chat(history).send_message(augmented_prompt)
  │       └── Gemini sees: rules + chat history + 5 chunks + question
  │
  ├── 5. VERIFY
  │     verify_citations(reply, context_text)
  │       └── Checks every quoted string against retrieved chunks
  │           Replaces unverified quotes with warning
  │
  ├── 6. SAVE SESSION
  │     sessions.save_history(session_id, updated_history)
  │       └── Writes data/sessions/<id>.json
  │
  └── Returns { reply, sources_used, chunks_found }
            │
            ▼
Student browser
  └── Displays answer with copy + feedback buttons
```

---

## 10. Remaining Limitations

Even with RAG, some limitations remain:

| Limitation | Explanation | Solution |
|---|---|---|
| Chunking cuts context | An 800-char chunk might cut a sentence in half | Better sentence-aware chunking |
| Low-score queries | If no chunk scores above 0.45, bot says "not found" — may be wrong | Lower threshold + fallback to context stuffing |
| Embedding cost | Every question requires an embedding API call (~50ms) | Cache recent query embeddings |
| Index not auto-refreshed | HUJI changes page → index is stale until restart | Scheduled daily refresh |
| Still can hallucinate | Gemini can paraphrase content incorrectly | Human review layer, feedback loop |
| Sessions are local files | Cannot share sessions across multiple servers | Redis for distributed deployment |

---

## Summary in One Paragraph

RAG solves the fundamental problem that LLMs are frozen in time and don't know private institutional knowledge. Instead of asking Gemini to answer from memory (wrong) or flooding it with all documents every time (wasteful), RAG converts every document into a mathematical representation of its meaning (embedding), stores those representations in a searchable index (FAISS), and at query time finds only the documents most relevant to the specific question. Gemini then answers from a small, precise, verified context — not from hallucination and not from irrelevant noise. The result is an accurate, cost-efficient, scalable system that grounds every answer in official sources.

---

*Document prepared: May 2026*  
*Stack: Python · FastAPI · Google Gemini 2.5 Flash · gemini-embedding-001 · FAISS · Playwright · pdfplumber*
