# 🧠 Smart Emotional ChatBot

An AI-powered mental health support chatbot built with a Retrieval-Augmented Generation (RAG) pipeline, Groq's LPU-accelerated LLMs, and a proactive crisis-alert system.

## Abstract

This project develops a mental health support chatbot utilizing Groq's LPU-powered LLMs and a Retrieval-Augmented Generation (RAG) pipeline. By leveraging a Chroma vector database and Hugging Face models, the system performs semantic searches across curated literature to provide empathetic, evidence-based responses. The application is deployed via a Gradio interface, facilitating personalized user interaction and enhanced mental health literacy. For user safety, the system incorporates a proactive alert mechanism that automatically notifies a trusted contact via email if patterns of distress are detected, ensuring reliable and secure mental health guidance. The knowledge base emphasizes key mental health topics including the definitions of mental health and illness, the scale of adolescent mental health problems, the impact of stigma, and evidence-based interventions.

## Features

- **RAG-based Q&A** — retrieves answers from curated mental health PDFs using Chroma + HuggingFace embeddings, grounding responses in real literature instead of the model's raw knowledge.
- **Crisis detection** — scans every user message for distress-related keywords.
- **Automated trusted-contact alerts** — on crisis detection, emails a pre-registered trusted contact with the user's details and immediate next steps.
- **In-chat crisis resources** — surfaces helpline numbers (India, US, international) directly to the user.
- **User authentication** — signup/login with hashed passwords and validated contact info.
- **Custom Gradio UI** — styled auth and chat views.

## Tech Stack

Python · LangChain · Groq API (Llama-3.3-70B) · ChromaDB · HuggingFace sentence-transformers · Gradio · SMTP

## Setup

1. Clone the repo and install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in your own values:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   GMAIL_SENDER_EMAIL=your_gmail_address_here
   GMAIL_APP_PASSWORD=your_gmail_app_password_here
   ```
   - Get a Groq API key from [console.groq.com](https://console.groq.com).
   - Generate a Gmail App Password from your Google Account → Security → App Passwords (requires 2-Step Verification enabled).

3. Load the `.env` file before running (e.g. with `python-dotenv`, or by exporting the variables in your shell).

4. Place your source PDF documents in a `data/` folder (update the path in `create_vector_db()` if needed).

5. Run the app:
   ```
   python app.py
   ```

## Notes

- `users.json` and `crisis_alerts.json` are created at runtime and are **not** included in this repo — they contain personal data and are excluded via `.gitignore`.
- The `chroma_db/` vector store is also excluded; it's regenerated automatically from the PDFs in `data/` on first run.
- This is a prototype/academic project and is not a substitute for professional mental health care.
