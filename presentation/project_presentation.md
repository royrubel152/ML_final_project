# MBA Academic Advisor Chatbot
### Hebrew University – Business School
#### Project Presentation | Development Summary & Timeline

---

## Table of Contents
1. [Project Brief & Client Requirements](#1-project-brief--client-requirements)
2. [Phase 1 — Core Architecture](#2-phase-1--core-architecture)
3. [Phase 2 — Data Enrichment & Intelligence](#3-phase-2--data-enrichment--intelligence)
4. [Phase 3 — UX, Quality & Advanced Features](#4-phase-3--ux-quality--advanced-features)

---

## 1. Project Brief & Client Requirements

### The Problem
MBA students and prospective applicants at the Hebrew University had no fast, reliable way to get answers about:
- Admission requirements and procedures
- Specialization tracks and course details
- Academic exemptions and regulations
- Accelerated and combined study tracks

Existing resources were fragmented across multiple official websites and PDFs. Students had to navigate several pages and a complex academic catalog (Shnaton) manually.

### What the Client Asked For
| Requirement | Description |
|---|---|
| Intelligent Q&A bot | Answer student questions in Hebrew based on official sources only |
| Official sources only | No hallucination — answers grounded in real HUJI documents |
| Personalized responses | Responses tailored to student type (active student, prospective, alumni) |
| Course & specialization data | Include teacher names, schedules, grade breakdowns |
| Study planner | Allow students to plan and visualize their semester schedule |
| Clean, professional UI | RTL Hebrew interface suitable for a university product |

### Success Criteria
- Bot answers must be grounded in official sources (no invented facts)
- Response quality must match what a human advisor would say
- Interface must be intuitive for non-technical students
- System must be maintainable and updatable as the curriculum changes

---

## 2. Phase 1 — Core Architecture
**Timeline: Weeks 1–3**

### What We Built
The foundational infrastructure: a full-stack AI system with a retrieval pipeline that grounds every answer in official content.

### Technology Stack
| Layer | Technology | Purpose |
|---|---|---|
| Backend | FastAPI + Uvicorn | Async API server |
| LLM | Google Gemini 2.5 Flash | Hebrew language generation |
| Embeddings | gemini-embedding-001 (3072 dims) | Semantic search |
| Vector DB | FAISS IndexFlatIP | Fast similarity retrieval |
| Scraping | Playwright + Requests | Fetching official web content |
| Frontend | HTML/CSS/JS (vanilla) | Lightweight, no framework needed |

### RAG Pipeline (Retrieval Augmented Generation)
This was the core architectural decision. Instead of asking Gemini questions directly (which risks hallucination), every query goes through a 5-step pipeline:

```
User Question
     ↓
1. Embed question (gemini-embedding-001)
     ↓
2. Search FAISS index → retrieve top-5 relevant chunks
     ↓
3. If no relevant chunks found → return "off-topic" reply
     ↓
4. Build augmented prompt: [system rules] + [retrieved context] + [question]
     ↓
5. Gemini generates answer grounded only in retrieved text
```

### Sources Scraped (Initial)
- HUJI Business School: academic regulations PDF, exemptions page, admissions, specializations, accelerated tracks
- All content chunked into ~300-word segments and embedded

### Key Design Decisions
- **Off-topic gate**: if no relevant chunks are retrieved, the bot refuses to answer rather than guessing
- **Citation verifier**: checks that quoted text actually appears in the source context
- **Session history**: last 10 conversation turns stored per user so follow-up questions work naturally
- **Rate limiting**: 15 requests/minute per IP to prevent abuse

### Outcome at End of Phase 1
- Working chatbot answering questions from 6 official HUJI pages
- ~50 indexed chunks
- Basic Hebrew chat UI with send/receive functionality

---

## 3. Phase 2 — Data Enrichment & Intelligence
**Timeline: Weeks 4–6**

### The Problem We Solved
The initial scraper only covered the main bschool.huji.ac.il pages. The actual course catalog lived on **shnaton.huji.ac.il** — a React single-page application (SPA) that couldn't be scraped with standard tools because it renders content via JavaScript API calls, not static HTML.

### Technical Discovery: Reverse-Engineering the Shnaton API
We used Playwright to intercept network requests while browsing the Shnaton and discovered three undocumented internal API endpoints:

| Endpoint | Data Provided |
|---|---|
| `/api/yearly-roadmaps/{code}/thresholds` | Full program roadmap with all mandatory courses |
| `/api/specializations/{code}/thresholds` | All courses in a specialization track |
| `/api/syllabus?courseCode={code}&year=2026` | Full syllabus: teacher, schedule, grade breakdown, description |

### Scale of Data Integration
| Source Type | Count | Description |
|---|---|---|
| MBA Roadmaps | 7 | All מוסמך programs (MBA, accelerated, combined tracks) |
| Specializations | 20 | All 20 specialization tracks |
| Syllabus API calls | Per course | Teacher name, meeting day/time, grade breakdown, office hours |

**Chunk count grew from ~50 → 405 indexed chunks**

### Rich Course Data Now Included
Every mandatory course now contains:
```
- Course code & name
- Credits (נ"ז) and semester
- Teacher name and title (e.g., פרופסור / ד"ר)
- Meeting schedule (day + time)
- Grade breakdown (final exam %, midterm %, assignment %)
- Course description
- Prerequisites
```

### Programs Integrated
| Code | Program |
|---|---|
| 943-3894 | MBA רגיל |
| 322-3220 | מסלול ישיר מהתואר הראשון |
| 322-3222 | מסלול מואץ |
| 826-3254 | MBA + תואר שני במשפטים |
| 826-3255 | MBA + M.Sc. |
| 322-8168 | תכנית אלפא |
| 322-3792 | מסלול מצטיינים |

### Outcome at End of Phase 2
- 405 chunks covering all MBA programs and specializations
- Bot can now answer questions like "מה שעות הקבלה של המרצה?" or "מה חלוקת הציונים בקורס?"
- Scraper caches content for 24 hours; stale cache used as fallback if scraping fails

---

## 4. Phase 3 — UX, Quality & Advanced Features
**Timeline: Weeks 7–10**

### 4.1 Quality Fixes

#### Problem: Broken Citation Verifier
The citation verifier was checking whether any quoted string appeared verbatim in the source text. Hebrew names like `ד"ר` contain a `"` character, causing the regex to match from `ד"ר` all the way to the next `"` in the answer — and flag it as unverified.

**Fix:** Raised the threshold to 100 characters. Only quotes longer than 100 characters are checked, eliminating false positives on Hebrew abbreviations and short names.

#### Problem: Raw Markdown in Chat
Bot responses used `**bold**` and tables, but the frontend rendered plain text — students saw asterisks and pipe characters.

**Fix:** Integrated `marked.js` (CDN) to render Markdown to HTML inside chat bubbles. Added CSS for styled tables, bold headings, and lists inside bot messages.

---

### 4.2 Onboarding Wizard
**What it does:** Before the first message, a step-by-step wizard collects student context. This context is silently prepended to every chat message so Gemini always knows who it's talking to.

**5-step flow:**
```
Step 1: Who are you?          → Active student / Prospective / Alumni
Step 2: What year?            → Year 1 / Year 2 / Not relevant
Step 3: Two specializations?  → Yes (major + minor) / No (single)
Step 4: Major specialization  → Pick from 12 specialization options
Step 5: Minor specialization  → (Only shown if answered Yes in Step 3)
```

**Profile prepended to every message:**
```
[פרופיל סטודנט: סטודנט פעיל ב-MBA | שנה א' | התמחות ראשית: מימון ובנקאות | התמחות משנית: פינטק]
```

**Technical note:** Steps are shown/hidden using inline `style.display` (not CSS classes) to guarantee visibility control regardless of stylesheet cascade order. Progress bar animates from 0% → 100% as steps complete.

---

### 4.3 Course Schedule Planner
**What it does:** A full-screen modal planner where students can build and visualize their 4-semester course schedule.

**Features:**
| Feature | Detail |
|---|---|
| 4-column layout | One column per semester (Year 1A, 1B, Year 2A, 2B) |
| Unlimited courses | Add as many courses per semester as needed |
| Live autocomplete | Start typing → dropdown shows matching courses from all 162 in the catalog |
| Remove courses | One-click removal per course |
| Clear all | Confirmation dialog before wiping the plan |
| Mobile responsive | Collapses to 2-column grid on small screens |

**Autocomplete implementation:**
- Backend `/courses` endpoint parses all 405 chunks, extracts course names via regex (`^\s*-\s+\d+:\s+(.+?)\s+\(`)
- Returns 162 unique course names
- Frontend fetches once on first modal open, caches in memory
- Filters client-side as user types, shows top 8 matches
- `mousedown` (not `click`) used for selection to prevent `blur` event race condition

---

## Summary Timeline (Completed Work)

| Week | Dates | Milestone |
|---|---|---|
| 1–2 | Early Apr | FastAPI server, Gemini integration, basic Hebrew chat UI |
| 3 | Mid Apr | RAG pipeline: FAISS index, embeddings, retrieval, off-topic gate |
| 4 | Late Apr | Playwright scraping of 6 bschool.huji.ac.il official pages |
| 5 | Early May | Reverse-engineered Shnaton API — 7 roadmaps + 20 specializations |
| 6 | Mid May | Syllabus enrichment per course (teacher, schedule, grades) → 405 chunks |
| 7 | Late May | Fixed citation verifier, markdown rendering, response formatting |
| 8 | 21–28 May | Onboarding wizard (5 steps, dual specialization), schedule planner modal |
| 9 | 28 May | Live autocomplete from 162 real courses, final UI polish |

---

## Progress Update for Ran — What We Built (3-4 Steps)

> **Summary for the business meeting: what we did, why it matters, and where we are now.**

### Step 1 — We Built the Foundation (a real AI advisor, not just a chatbot)
We built a full AI system on top of Google's Gemini model, but with a critical difference from a standard chatbot: **every answer is grounded in official HUJI sources only**. We scrape the official MBA pages automatically, chunk the content, embed it into a vector database (FAISS), and retrieve the most relevant pieces before every answer. If the question is off-topic or has no matching content, the bot refuses to answer rather than guessing. This eliminates hallucination.

### Step 2 — We Cracked the Shnaton (the hard part)
The full course catalog lives on shnaton.huji.ac.il — a React app that doesn't expose a public API. We reverse-engineered its internal network calls and discovered 3 undocumented API endpoints. This let us pull **all 27 MBA programs and specializations** directly, including teacher names, class schedules, grade breakdowns, and course descriptions. The knowledge base grew from ~50 chunks to **405 indexed chunks** covering the entire MBA curriculum.

### Step 3 — We Made It Personal (student profiling)
Instead of giving generic answers, the bot now knows who it's talking to. An onboarding wizard at startup asks 5 questions: are you an active student or prospective? What year? Do you have one or two specializations? Every subsequent message silently includes this profile — so when a Year 1 Finance student asks about exemptions, the answer is tailored to them specifically.

### Step 4 — We Added a Study Planner (beyond just Q&A)
We built a visual course schedule planner inside the app: a full-screen modal with 4 semester columns where students can search and add courses from the live 162-course catalog via autocomplete. This transforms the tool from a Q&A bot into an actual academic planning assistant — something Gemini or ChatGPT alone cannot offer.

---

## Remaining Schedule — May 28 to June 22

| Week | Dates | Focus | Deliverables |
|---|---|---|---|
| Week 1 | May 28 – Jun 1 | **Evaluation & testing** | Run 20+ test questions across all specializations. Log gaps where bot answers weakly or incorrectly. |
| Week 2 | Jun 2 – Jun 8 | **Fine-tuning & fixes** | Improve system prompt based on test results. Fix any weak answer patterns. Re-scrape if content gaps found. |
| Week 3 | Jun 9 – Jun 15 | **Business meeting with Ran + polish** | Present progress update. Collect feedback. Apply final UI/UX improvements based on feedback. |
| Week 4 | Jun 16 – Jun 22 | **Final delivery** | Full demo preparation. Documentation complete. Final deployment-ready build. **Deadline: June 22** |

### Week-by-Week Detail

#### May 28 – June 1 | Evaluation
- Define 20 representative test questions (admissions, exemptions, course details, schedules, specialization requirements)
- Run each question through the bot and grade: correct / partially correct / wrong
- Identify which specializations or topics have weak coverage
- Document all failure cases

#### June 2 – June 8 | Fine-Tuning
- Refine the system prompt based on failure patterns observed
- If coverage gaps found → update scraper to pull missing sources
- Improve response formatting for edge cases (e.g., multi-part questions)
- Re-test all previously failed questions

#### June 9 – June 15 | Business Meeting with Ran
- Prepare a short demo (use the Live Demo Checklist below)
- Walk through the 4-step progress summary above
- Show before/after: what a student had to do before vs. now
- Collect any final feature requests or adjustments
- Implement prioritized feedback from meeting

#### June 16 – June 22 | Final Delivery
- Freeze new features — focus on stability
- Final round of testing
- Clean up code and documentation
- Prepare final presentation slides
- **Submit by June 22**

---

## Key Technical Achievements

1. **Reverse-engineered a React SPA** to extract structured academic data that had no public API documentation
2. **Zero hallucination architecture** — off-topic gate + citation verifier ensure every answer is traceable to a source
3. **Full Hebrew RTL support** throughout — both the UI and the LLM respond exclusively in Hebrew
4. **Session-aware conversations** — history preserved per user so follow-up questions work naturally
5. **Production-ready features** — rate limiting, error handling, 24h content cache with stale fallback, health endpoint

---

## Live Demo Checklist
- [ ] Show onboarding wizard (5 steps, dual specialization path)
- [ ] Ask: "מה תנאי הקבלה לתוכנית MBA?"
- [ ] Ask: "מי מרצה בקורס פינטק ומתי הוא מתקיים?"
- [ ] Ask a question about weather (off-topic gate)
- [ ] Open schedule planner → add courses via autocomplete
- [ ] Show "שיחה חדשה" reset flow
