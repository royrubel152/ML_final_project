from fpdf import FPDF, XPos, YPos

BLUE  = (0, 63, 127)
GOLD  = (232, 160, 32)
LIGHT = (244, 246, 251)
GRAY  = (107, 114, 128)
WHITE = (255, 255, 255)
DARK  = (26, 26, 46)


def clean(text):
    return (text
        .replace("—", "-")
        .replace("–", "-")
        .replace("‘", "'")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("…", "...")
    )


class PDF(FPDF):
    def header(self):
        self.set_fill_color(*BLUE)
        self.rect(0, 0, 210, 18, "F")
        self.set_y(4)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*WHITE)
        self.cell(0, 10, "MBA Academic Advisor Bot  |  Hebrew University Business School",
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*DARK)
        self.ln(10)

    def footer(self):
        self.set_y(-14)
        self.set_draw_color(*BLUE)
        self.set_line_width(0.4)
        self.line(15, self.get_y(), 195, self.get_y())
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 8,
            f"Page {self.page_no()}  |  Confidential - Hebrew University Business School  |  May 2026",
            align="C")

    def cover_page(self):
        self.set_fill_color(*BLUE)
        self.rect(0, 0, 210, 80, "F")
        self.set_fill_color(*GOLD)
        self.rect(0, 80, 210, 4, "F")

        self.set_y(20)
        self.set_font("Helvetica", "B", 26)
        self.set_text_color(*WHITE)
        self.cell(0, 12, "MBA Academic Advisor Bot",
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_font("Helvetica", "", 14)
        self.cell(0, 10, "Hebrew University Business School",
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_font("Helvetica", "", 10)
        self.set_text_color(200, 220, 255)
        self.cell(0, 8, "Client Documentation & Cost Analysis",
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_y(100)
        self.set_fill_color(*LIGHT)
        self.set_draw_color(*BLUE)
        self.set_line_width(0.5)
        self.rect(30, 100, 150, 58, "FD")

        self.set_y(108)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*BLUE)
        self.cell(0, 8, "Document Overview",
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        items = [
            "System architecture & technology stack",
            "Features & capabilities",
            "Step-by-step usage guide",
            "Full cost breakdown (current & scaled)",
            "Future improvement roadmap",
        ]
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*DARK)
        for item in items:
            self.set_x(42)
            self.set_text_color(*GOLD)
            self.cell(6, 7, ">", new_x=XPos.RIGHT, new_y=YPos.TOP)
            self.set_text_color(*DARK)
            self.cell(0, 7, item, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_y(170)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*GRAY)
        self.cell(0, 6, "Prepared: May 2026", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(0, 6, "Technology: Python + Google Gemini AI", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(0, 6, "Current monthly cost: $0", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.add_page()

    def section_title(self, number, title):
        self.ln(4)
        self.set_fill_color(*BLUE)
        self.rect(15, self.get_y(), 4, 8, "F")
        self.set_x(22)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*BLUE)
        self.cell(0, 8, f"{number}. {title}",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*GOLD)
        self.set_line_width(0.4)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(4)
        self.set_text_color(*DARK)

    def sub_title(self, title):
        self.ln(2)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*BLUE)
        self.cell(0, 7, clean(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*DARK)

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*DARK)
        self.set_x(15)
        self.multi_cell(180, 6, clean(text))
        self.ln(1)

    def bullet(self, text, indent=20):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*DARK)
        self.set_x(indent)
        self.cell(5, 6, "-", new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.multi_cell(175 - indent, 6, clean(text))

    def info_box(self, text):
        self.ln(2)
        self.set_fill_color(*LIGHT)
        self.set_draw_color(*BLUE)
        self.set_line_width(0.3)
        self.set_x(15)
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*DARK)
        self.multi_cell(180, 6, clean(text), border=1, fill=True)
        self.ln(2)

    def table_header(self, cols, widths):
        self.set_fill_color(*BLUE)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 9)
        self.set_x(15)
        for i, (col, w) in enumerate(zip(cols, widths)):
            nx = XPos.RIGHT if i < len(cols) - 1 else XPos.LMARGIN
            ny = YPos.TOP if i < len(cols) - 1 else YPos.NEXT
            self.cell(w, 7, col, border=1, fill=True, new_x=nx, new_y=ny)
        self.set_text_color(*DARK)

    def table_row(self, cells, widths, fill=False):
        self.set_fill_color(244, 246, 251) if fill else self.set_fill_color(*WHITE)
        self.set_font("Helvetica", "", 9)
        self.set_x(15)
        for i, (cell, w) in enumerate(zip(cells, widths)):
            nx = XPos.RIGHT if i < len(cells) - 1 else XPos.LMARGIN
            ny = YPos.TOP if i < len(cells) - 1 else YPos.NEXT
            self.cell(w, 6, clean(str(cell)), border=1, fill=fill, new_x=nx, new_y=ny)

    def cost_highlight(self, label, value, note=""):
        self.ln(2)
        self.set_fill_color(*BLUE)
        self.set_x(15)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*WHITE)
        self.cell(100, 8, clean(label), fill=True, border=0, new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.set_fill_color(*GOLD)
        self.cell(80, 8, clean(value), fill=True, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if note:
            self.set_x(15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(*GRAY)
            self.cell(0, 5, clean(note), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*DARK)
        self.ln(1)


def generate():
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(15, 22, 15)
    pdf.add_page()
    pdf.cover_page()

    # 1 - What Was Built
    pdf.section_title(1, "What Was Built")
    pdf.body(
        "A web-based AI chatbot that acts as an academic advisor for students in the MBA program "
        "at the Hebrew University Business School. Students open a link in their browser on any "
        "device (phone, computer, tablet), type a question in Hebrew, and receive a precise answer "
        "based exclusively on official university sources."
    )
    pdf.body(
        "The system was designed to reduce the volume of repetitive questions directed at the "
        "student secretariat, provide instant 24/7 responses, and ensure that all answers are "
        "grounded in verified, official academic information."
    )

    # 2 - How It Works
    pdf.section_title(2, "How It Works")
    pdf.info_box(
        "Student types question  >>  Chat interface (website)  >>  Python server\n"
        ">>  Google Gemini AI (reads official HUJI content)  >>  Structured Hebrew answer"
    )
    pdf.body(
        "At every server startup, the system automatically fetches the latest content from all "
        "6 official HUJI sources (web pages + PDF). This content is injected into the AI model's "
        "context. The AI is strictly instructed to answer only from this content - it cannot "
        "invent or guess information."
    )

    # 3 - Official Sources
    pdf.section_title(3, "Official Sources")
    pdf.body("The bot reads and answers exclusively from the following 6 official HUJI sources:")
    pdf.ln(2)
    pdf.table_header(["Source", "URL"], [65, 115])
    rows = [
        ("Academic Regulations (PDF)", "bschool.huji.ac.il - Academic Regulations PDF"),
        ("Exemptions", "bschool.huji.ac.il/ptorim"),
        ("Admissions", "bschool.huji.ac.il/mba/admittance"),
        ("Specializations", "bschool.huji.ac.il/mba/specializations"),
        ("Accelerated Track", "bschool.huji.ac.il/Accelerated-study-track"),
        ("Accelerated Track - All Courses", "bschool.huji.ac.il/Accelerated-study-track-all"),
    ]
    for i, row in enumerate(rows):
        pdf.table_row(row, [65, 115], fill=(i % 2 == 0))
    pdf.ln(3)

    # 4 - Capabilities
    pdf.section_title(4, "Capabilities & Limitations")
    pdf.sub_title("What the bot CAN answer:")
    for item in [
        "Admissions requirements and application process",
        "Available specializations and their structure",
        "Exemption requests - eligibility and process",
        "Accelerated study track - who qualifies, how to apply",
        "Academic regulations - credits, grades, course requirements",
        "General MBA program structure and rules",
    ]:
        pdf.bullet(item)
    pdf.ln(3)
    pdf.sub_title("What the bot CANNOT answer:")
    for item in [
        "Questions not covered in the 6 official sources",
        "Personal student account issues (grades, registration systems)",
        "Real-time scheduling or current course availability",
        "Anything requiring a human administrative decision",
    ]:
        pdf.bullet(item)
    pdf.ln(2)
    pdf.info_box(
        "In all unsupported cases, the bot directs the student to contact the student secretariat."
    )

    # 5 - Features
    pdf.section_title(5, "Features")
    pdf.table_header(["Feature", "Description"], [50, 130])
    features = [
        ("Hebrew UI", "Full right-to-left interface, designed for Hebrew speakers"),
        ("Source links", "Sidebar with direct clickable links to all 6 official sources"),
        ("Quick questions", "One-click buttons for the 4 most common student topics"),
        ("Chat history", "Remembers context within a session for follow-up questions"),
        ("Reset button", "Start a fresh conversation at any time"),
        ("Mobile friendly", "Fully responsive - works on phones and tablets"),
        ("Disclaimer", "Every answer includes a reminder to verify with secretariat"),
        ("Typing indicator", "Shows animation while the AI is generating a response"),
        ("Timestamps", "Each message shows the time it was sent"),
        ("Error handling", "Clear messages if the server or API is unavailable"),
    ]
    for i, row in enumerate(features):
        pdf.table_row(row, [50, 130], fill=(i % 2 == 0))
    pdf.ln(3)

    # 6 - Tech Stack
    pdf.section_title(6, "Technology Stack")
    pdf.table_header(["Component", "Technology", "Reason"], [45, 55, 80])
    tech = [
        ("Backend", "Python + FastAPI", "Fast, reliable, easy to maintain"),
        ("AI Model", "Google Gemini 2.5 Flash", "Best Hebrew support, fast responses"),
        ("Content loader", "BeautifulSoup + pdfplumber", "Fetches real content from HUJI at startup"),
        ("Frontend", "HTML + CSS + JavaScript", "No framework needed, loads instantly"),
        ("Web server", "Uvicorn", "Industry-standard Python ASGI server"),
        ("Sharing", "ngrok", "Instant public URL for local server"),
    ]
    for i, row in enumerate(tech):
        pdf.table_row(row, [45, 55, 80], fill=(i % 2 == 0))
    pdf.ln(3)

    # 7 - How to Use
    pdf.section_title(7, "How to Use - Step by Step")
    pdf.sub_title("Requirements")
    for item in [
        "A computer with Python 3.10 or higher installed",
        "Internet connection (to reach HUJI sources and Gemini API)",
        "Google Gemini API key (already configured in the project)",
    ]:
        pdf.bullet(item)
    pdf.ln(3)

    pdf.sub_title("Starting the server")
    pdf.table_header(["Step", "Action"], [25, 155])
    steps = [
        ("1", "Open a terminal (Command Prompt or PowerShell)"),
        ("2", "Navigate to project: cd \"C:\\...\\final_project_big\""),
        ("3", "Run: uvicorn app:app --reload"),
        ("4", "Wait for: [startup] Done. Total content: ~31000 chars"),
        ("5", "Open browser at: http://localhost:8000"),
    ]
    for i, row in enumerate(steps):
        pdf.table_row(row, [25, 155], fill=(i % 2 == 0))
    pdf.ln(3)

    pdf.sub_title("Sharing with students (ngrok)")
    for item in [
        "Keep uvicorn running in the first terminal",
        "Open a second terminal and run: ngrok http 8000",
        "Copy the public link (e.g. https://abc123.ngrok-free.app)",
        "Send the link to students via WhatsApp, email, or the university portal",
        "Note: The link changes every time ngrok restarts. The computer must stay on.",
    ]:
        pdf.bullet(item)
    pdf.ln(3)
    pdf.sub_title("Stopping the server")
    pdf.body("Press CTRL + C in the terminal window where uvicorn is running.")

    # 8 - Costs
    pdf.section_title(8, "Cost Breakdown")
    pdf.sub_title("Current Setup - Local + ngrok (running today)")
    pdf.table_header(["Item", "Plan", "Monthly Cost"], [75, 65, 40])
    for i, row in enumerate([
        ("Google Gemini 2.5 Flash API", "Free tier (1,500 req/day)", "$0"),
        ("ngrok", "Free tier (1 public URL)", "$0"),
        ("Hosting", "Local computer", "$0"),
    ]):
        pdf.table_row(row, [75, 65, 40], fill=(i % 2 == 0))
    pdf.ln(2)
    pdf.cost_highlight(
        "Total current monthly cost:",
        "$0",
        "Limitation: computer must stay on and connected for the bot to work."
    )

    pdf.ln(4)
    pdf.sub_title("Cloud Deployment (recommended for full student access - 24/7)")
    pdf.body("If the university wants the bot always online for all students:")
    pdf.table_header(["Item", "Option", "Monthly Cost"], [65, 75, 40])
    for i, row in enumerate([
        ("Cloud hosting", "Railway.app (Starter)", "~$5"),
        ("Cloud hosting", "Render.com (free tier)", "$0 (limited)"),
        ("Cloud hosting", "Google Cloud Run", "$5-$15"),
        ("Domain name (optional)", "Custom URL", "~$1-$2"),
        ("Gemini API (free tier)", "Up to 1,500 req/day", "$0"),
        ("Gemini API (paid)", "Above free tier", "~$0.15/1M tokens"),
    ]):
        pdf.table_row(row, [65, 75, 40], fill=(i % 2 == 0))
    pdf.ln(2)
    pdf.cost_highlight(
        "Estimated cost for ~50 students/day:",
        "$5-$10 / month",
        "Includes hosting + Gemini API at moderate usage."
    )

    pdf.ln(4)
    pdf.sub_title("Gemini API Pricing Detail")
    pdf.table_header(["Model", "Input tokens", "Output tokens"], [70, 60, 60])
    for i, row in enumerate([
        ("Gemini 2.5 Flash (free)", "1,500 req/day free", "1,500 req/day free"),
        ("Gemini 2.5 Flash (paid)", "$0.15 per 1M tokens", "$0.60 per 1M tokens"),
    ]):
        pdf.table_row(row, [70, 60, 60], fill=(i % 2 == 0))
    pdf.ln(2)
    pdf.info_box(
        "Each student conversation uses approx. 5,000-8,000 tokens (full source content is included "
        "in every request). At paid rates: 100 conversations = approx. $0.05-$0.08 total."
    )

    # 9 - Important Notes
    pdf.section_title(9, "Important Notes")
    for title, text in [
        ("Content refresh", "Content is fetched at startup only. If HUJI updates their website, restart the server to reload."),
        ("Not a replacement", "The bot supplements - not replaces - the student secretariat. Every answer includes a disclaimer."),
        ("Session memory", "Conversation history is temporary and lost when the server restarts."),
        ("API key security", "The .env file contains the API key. Never share it or upload it to GitHub."),
        ("PDF parsing", "The academic regulations PDF is parsed automatically. A server restart reloads updated content."),
    ]:
        pdf.sub_title(title)
        pdf.body(text)

    # 10 - Roadmap
    pdf.section_title(10, "Future Improvements & Roadmap")
    pdf.table_header(["Feature", "Effort", "Est. Cost"], [105, 30, 45])
    for i, row in enumerate([
        ("Deploy to cloud (always online, no local computer needed)", "Low", "~$5/month"),
        ("WhatsApp / Telegram integration", "Medium", "Provider cost"),
        ("Admin panel to update sources manually", "Medium", "Dev cost only"),
        ("Student login and personalization", "High", "Dev cost only"),
        ("Analytics dashboard (topics, usage, frequency)", "Medium", "Dev cost only"),
        ("Persistent chat history per student", "Medium", "~$0 DB cost"),
        ("Multi-language support (English)", "Low", "Dev cost only"),
    ]):
        pdf.table_row(row, [105, 30, 45], fill=(i % 2 == 0))

    # 11 - Contact
    pdf.ln(4)
    pdf.section_title(11, "Contact & Support")
    pdf.body(
        "For technical questions or to request upgrades, contact the development team.\n"
        "For content accuracy questions, contact the MBA program secretariat at the "
        "Hebrew University Business School."
    )
    pdf.ln(2)
    pdf.info_box(
        "This document was prepared at the end of the initial development phase - May 2026.\n"
        "All costs are estimates based on current provider pricing and may change."
    )

    pdf.output("CLIENT_DOCUMENTATION.pdf")
    print("PDF saved: CLIENT_DOCUMENTATION.pdf")


if __name__ == "__main__":
    generate()
