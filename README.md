# LearnLens — AI-Powered Exam Intelligence Platform 📘

> Built during the Sprint4Good AI Hackathon at IIT Delhi in just 12 hours by two first-year students.

LearnLens is an AI-powered study platform that lets students upload their own notes and interact with them intelligently. Instead of getting generic internet answers, LearnLens responds strictly from your uploaded syllabus notes using Retrieval-Augmented Generation (RAG).

The goal was simple:

Students waste hours searching random websites or asking AI tools questions that don’t align with their actual syllabus. LearnLens solves this by creating a personal AI tutor trained only on your notes.

# ✨ Features

## 📤 Upload Your Notes
- Upload PDF notes directly
- Automatic text extraction using PyMuPDF
- Smart chunking + vector embeddings
- Notes stored in MongoDB Atlas Vector Search

---

## 💬 Ask Questions from Your Notes
- Ask questions in English or Hinglish
- AI searches only inside your uploaded notes
- Returns contextual answers with page references
- Prevents hallucinated internet-style answers

Example:
> “Explain Kirchhoff’s Law”
→ LearnLens searches your uploaded notes and answers only from them.

---

## 🧠 AI Summary Generator
Generate:
- Structured chapter summaries
- Key concepts
- Important definitions
- Formula revision points
- Quick exam revision notes

Perfect for last-minute preparation.

---

## ❓ Smart Quiz Generator
Generate:
- 10 MCQs instantly from notes
- Live score tracking
- Explanations for wrong answers
- Multiple difficulty levels:
  - Easy
  - Medium
  - Hard

---

## 📄 PYQ-Based Quiz Mode

Upload a Previous Year Question (PYQ) paper and LearnLens:
1. Analyzes the exam pattern
2. Understands the MCQ style
3. Generates new MCQs from your notes in the same pattern

# 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | Python |
| LLM | Ollama (`gemma3:4b`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Database | MongoDB Atlas Vector Search |
| PDF Parsing | PyMuPDF |
| Vector Search | MongoDB `$vectorSearch` |

# ⚙️ How LearnLens Works

## 1. PDF Ingestion
When a PDF is uploaded:
- Text is extracted page-by-page
- Notes are cleaned and processed
- Text is split into overlapping chunks
- Embeddings are generated using MiniLM
- Stored inside MongoDB Atlas

---

## 2. Semantic Search (RAG)
When the user asks a question:
- The query is converted into a vector embedding
- MongoDB Vector Search retrieves the most relevant chunks
- Retrieved context is passed to the LLM
- The LLM generates an answer strictly from the notes

---

## 3. Quiz Generation
For quizzes:
- Relevant note chunks are collected
- Prompt-engineered MCQs are generated
- JSON validation + parsing ensures structure consistency
- Interactive quiz UI evaluates answers live

# 📂 Project Structure

LearnLens/
│
├── LearnLens.py        # Main Streamlit application
├── uploads/            # Uploaded PDFs (auto-created)
├── README.md

# 🚀 Setup Guide

## 1. Clone the Repository

git clone https://github.com/your-username/LearnLens.git
cd LearnLens

---

## 2. Install Dependencies

pip install streamlit pymongo sentence-transformers ollama pymupdf

---

## 3. Install and Run Ollama

Download Ollama from:
https://ollama.com

Pull the model:

ollama pull gemma3:4b

---

## 4. Configure MongoDB Atlas

Replace this line inside `LearnLens.py`:

MONGO_URI = "your-mongodb-uri"

with your actual MongoDB connection string.

---

## 5. Create Vector Search Index

Create a Vector Search Index with:

{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    }
  ]
}

Index Name:
vector_index

---

## 6. Run the Application

streamlit run LearnLens.py

# 🧪 Current Limitations

Since this was built during a 12-hour hackathon, there are still many things we want to improve:

- Multi-user authentication
- Better UI/UX
- Multiple PDF support in one session
- Cloud deployment
- Flashcard generation
- Chat history
- Faster retrieval pipeline
- Better citation formatting
- Mobile responsiveness

# 🎯 Hackathon Story

LearnLens was built during the Sprint4Good AI Hackathon at IIT Delhi.

We were a team of just two first-year students competing against much larger and more experienced teams. Everything — from the idea to the working prototype — was built within 12 hours.

Although we did not make it to the Top 6 teams, one moment during the hackathon genuinely motivated us.

While testing our “Ask Questions” feature, a mentor compared LearnLens side-by-side with ChatGPT Live using syllabus-based questions. Since LearnLens answered strictly from uploaded notes using RAG, the responses turned out to be significantly more syllabus-aligned and exam-focused.

That validation made us realize the real potential of personalized AI learning systems.


# 📸 Future Vision

We want LearnLens to evolve into a complete AI exam-preparation ecosystem where students can:
- Learn from personalized AI tutors
- Practice adaptive quizzes
- Generate smart revision material
- Analyze PYQ trends
- Build subject-wise preparation dashboards

# 👨‍💻 Authors

- Ojaswi Gautam
- NB Ritesh Varshan

Team APEX  
Sprint4Good AI Hackathon — IIT Delhi

# 📌 Note

This is the first hackathon version (V1) of LearnLens.

The focus was rapid prototyping, core functionality, and proving the concept under intense time constraints during the Sprint4Good AI Hackathon at IIT Delhi.

We are currently planning and improving LearnLens V2 with:
- Better UI/UX
- Multi-user support
- Faster retrieval pipeline
- Cloud deployment
- Advanced quiz intelligence
- Flashcards & revision modes
- Improved scalability and performance

This project is just the beginning.
