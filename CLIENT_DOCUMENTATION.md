# MBA Academic Advisor Bot — Client Documentation
### Hebrew University Business School
**Date:** May 2026  
**Prepared by:** Development Team  

---

## 1. What Was Built

A web-based AI chatbot that acts as an academic advisor for students in the MBA program at the Hebrew University Business School.

Students open a link in their browser (on any device — phone, computer, tablet), type a question in Hebrew, and receive an answer based **exclusively** on official university sources.

---

## 2. How It Works — Simple Explanation

```
Student types question
        ↓
Chat interface (website)
        ↓
Python server (runs on a computer)
        ↓
Google Gemini AI (reads the official HUJI content)
        ↓
Answer appears in Hebrew, formatted and sourced
```

At every startup, the system automatically fetches the latest content from all 6 official HUJI sources and loads it into the AI's memory. The AI is instructed to answer **only** from that content — it cannot invent or guess.

---

## 3. Official Sources the Bot Uses

| Source | URL |
|---|---|
| Academic Regulations (PDF) | bschool.huji.ac.il — תקנון אקדמי |
| Exemptions | bschool.huji.ac.il/ptorim |
| Admissions | bschool.huji.ac.il/mba/admittance |
| Specializations | bschool.huji.ac.il/mba/specializations |
| Accelerated Track | bschool.huji.ac.il/Accelerated-study-track |
| Accelerated Track — All Courses | bschool.huji.ac.il/Accelerated-study-track-all |

---

## 4. What the Bot Can Answer

- Admissions requirements and process
- Available specializations and their structure
- Exemption requests — eligibility and process
- Accelerated study track — who qualifies, how to apply
- Academic regulations — credits, grades, course requirements
- General MBA program structure and rules

## 5. What the Bot Cannot Answer

- Questions not covered in the 6 official sources
- Personal student account issues (grades, registration)
- Real-time scheduling or course availability
- Anything requiring a human decision

In all these cases, the bot directs the student to the student secretariat.

---

## 6. Features

| Feature | Description |
|---|---|
| Hebrew UI | Full right-to-left interface |
| Source links | Sidebar with direct links to all official sources |
| Quick questions | One-click buttons for the 4 most common topics |
| Chat history | Remembers context within a session |
| Reset button | Start a new conversation at any time |
| Mobile friendly | Works on phones and tablets |
| Disclaimer | Every answer includes a reminder to verify with secretariat |
| Typing indicator | Shows animation while AI is thinking |
| Timestamps | Each message shows the time it was sent |

---

## 7. Technology Stack

| Component | Technology | Why |
|---|---|---|
| Backend | Python + FastAPI | Fast, reliable, easy to maintain |
| AI Model | Google Gemini 2.5 Flash | Best Hebrew language support, fast responses |
| Content loading | Custom web scraper (BeautifulSoup + pdfplumber) | Fetches real content from HUJI sources at startup |
| Frontend | HTML + CSS + JavaScript | No framework needed, loads instantly |
| Server | Uvicorn (ASGI) | Industry standard Python web server |

---

## 8. Project File Structure

```
final_project_big/
├── app.py               ← Main server + AI logic
├── scraper.py           ← Fetches content from HUJI sources
├── templates/
│   └── index.html       ← Chat interface
├── static/
│   ├── style.css        ← Design and layout
│   └── script.js        ← Chat behavior
├── requirements.txt     ← Python dependencies
└── .env                 ← API key (private, never share)
```

---

## 9. How to Run — Step by Step

### Requirements
- A computer with Python 3.10 or higher installed
- Internet connection (to reach HUJI sources and Gemini API)
- A Google Gemini API key (already configured)

### Start the bot

**Step 1** — Open a terminal (Command Prompt or PowerShell)

**Step 2** — Navigate to the project folder:
```
cd "C:\Users\rubro\OneDrive\Desktop\final_project_big"
```

**Step 3** — Start the server:
```
uvicorn app:app --reload
```

**Step 4** — Wait until you see:
```
[startup] Done. Total content: ~31000 chars
INFO: Application startup complete.
```

**Step 5** — Open browser and go to:
```
http://localhost:8000
```

### Share with students (using ngrok)

**Step 1** — Keep uvicorn running  
**Step 2** — Open a second terminal window  
**Step 3** — Run:
```
ngrok http 8000
```
**Step 4** — Copy the public link (e.g. `https://abc123.ngrok-free.app`) and share it  

> The link changes every time you restart ngrok. The computer must stay on.

### Stop the server
Press `CTRL + C` in the terminal.

---

## 10. Costs

### Current Setup — Local + ngrok (what is running now)

| Item | Cost |
|---|---|
| Google Gemini API | **Free** — Gemini 2.5 Flash has a free tier (15 requests/min, 1,500 req/day) |
| ngrok | **Free** — for 1 public URL, limited to 1 concurrent session |
| Hosting | **Free** — runs on your own computer |
| **Total per month** | **$0** |

> **Limitation:** The computer must be on and connected for the bot to work. Not suitable for large-scale use.

---

### For a Proper Deployment (recommended if scaling up)

If the university wants the bot available 24/7 to all students without a computer running locally:

| Item | Option | Monthly Cost |
|---|---|---|
| Cloud hosting | Railway.app (Starter) | ~$5/month |
| Cloud hosting | Render.com (free tier) | $0 (with limitations) |
| Cloud hosting | Google Cloud Run | ~$5–15/month depending on usage |
| Domain name (optional) | e.g. mba-advisor.huji.ac.il | ~$1–2/month |
| Gemini API (free tier) | Up to 1,500 req/day | $0 |
| Gemini API (paid) | Above free tier | ~$0.15 per 1M input tokens |

**Realistic monthly cost for ~50 students/day:** approximately **$5–10/month** total.

---

### Gemini API Pricing Detail (Google)

| Model | Input | Output |
|---|---|---|
| Gemini 2.5 Flash (free tier) | 1,500 requests/day free | 1,500 requests/day free |
| Gemini 2.5 Flash (paid) | $0.15 per 1M tokens | $0.60 per 1M tokens |

> Each student conversation = roughly 5,000–8,000 tokens (because the full source content is included in every request). At paid rates, 100 conversations ≈ **$0.05–0.08**.

---

## 11. Limitations & Important Notes

1. **Content is fetched at startup only** — if HUJI updates their website, restart the server to reload the latest content.
2. **The bot is not a replacement for the secretariat** — every answer includes a disclaimer.
3. **Session memory is temporary** — conversation history is lost when the server restarts.
4. **API key is secret** — the `.env` file must never be shared or uploaded to GitHub.
5. **Hebrew PDF parsing** — the academic regulations PDF is parsed automatically. If the PDF is updated by HUJI, a server restart is sufficient to reload it.

---

## 12. Possible Future Improvements

| Feature | Effort | Cost |
|---|---|---|
| Deploy to cloud (always online) | Low | ~$5/month |
| WhatsApp / Telegram integration | Medium | Depends on provider |
| Admin panel to update sources manually | Medium | Development cost only |
| Student login / personalization | High | Development cost only |
| Analytics dashboard (questions asked, topics) | Medium | Development cost only |
| Persistent chat history per student | Medium | Small DB cost (~$0) |

---

## 13. Contact & Support

For technical questions or to request upgrades, contact the development team.  
For content accuracy questions, contact the MBA program secretariat at the Hebrew University Business School.

---

*This document was prepared at the end of the initial development phase — May 2026.*
