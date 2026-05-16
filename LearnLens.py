import os
import re
import uuid
import json
import streamlit as st
import fitz  # PyMuPDF
from datetime import datetime

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import ollama


# ==========================
# CONFIG
# ==========================
st.set_page_config(page_title="LearnLens", page_icon="📘", layout="wide")

MONGO_URI = "your-mongodb-uri"
DB_NAME = "LearnLens_db"
COLLECTION_NAME = "chunks"
VECTOR_INDEX_NAME = "vector_index"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "gemma3:4b"


# ==========================
# SESSION STATE INIT
# ==========================
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None

if "quiz_pdf_id" not in st.session_state:
    st.session_state.quiz_pdf_id = None

if "quiz_checked" not in st.session_state:
    st.session_state.quiz_checked = {}

if "quiz_result" not in st.session_state:
    st.session_state.quiz_result = {}

if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0

# quiz mode flow
if "show_quiz_mode" not in st.session_state:
    st.session_state.show_quiz_mode = False

if "quiz_mode" not in st.session_state:
    st.session_state.quiz_mode = None  # "notes" or "notes_pyq"

# quiz difficulty
if "quiz_difficulty" not in st.session_state:
    st.session_state.quiz_difficulty = "Medium"


# ==========================
# HELPERS
# ==========================
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_mongo_collection():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db[COLLECTION_NAME]


def clean_text(text: str) -> str:
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pages_from_pdf(pdf_path: str):
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        page_text = doc[i].get_text("text")
        page_text = clean_text(page_text)
        if page_text:
            pages.append({"page": i + 1, "text": page_text})
    doc.close()
    return pages


def chunk_text(text: str, chunk_size_words=220, overlap_words=40):
    words = text.split()
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(words):
        end = start + chunk_size_words
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append({"chunk_index": chunk_index, "chunk_text": chunk})
            chunk_index += 1

        start += (chunk_size_words - overlap_words)

    return chunks


def save_uploaded_pdf(uploaded_file):
    os.makedirs("uploads", exist_ok=True)
    pdf_id = str(uuid.uuid4())
    filename = uploaded_file.name.replace(" ", "_")
    path = f"uploads/{pdf_id}_{filename}"

    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return pdf_id, filename, path


def delete_previous_user_pdfs(col, user_id):
    result = col.delete_many({"user_id": user_id})
    return result.deleted_count


def store_chunks_in_mongo(pdf_id, filename, user_id, pages, col, model):
    inserted = 0

    for page_data in pages:
        page_no = page_data["page"]
        text = page_data["text"]

        chunks = chunk_text(text)

        for c in chunks:
            chunk_text_ = c["chunk_text"]
            embedding = model.encode(chunk_text_).tolist()

            doc = {
                "user_id": user_id,
                "pdf_id": pdf_id,
                "pdf_name": filename,
                "page": page_no,
                "chunk_index": c["chunk_index"],
                "chunk_text": chunk_text_,
                "embedding": embedding,
                "created_at": datetime.utcnow()
            }

            col.insert_one(doc)
            inserted += 1

    return inserted


# ✅ FIXED VECTOR SEARCH (no filter inside vectorSearch)
def vector_search(question: str, col, model, user_id, top_k=4):
    q_emb = model.encode(question).tolist()

    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": q_emb,
                "numCandidates": 120,
                "limit": top_k * 6,
            }
        },
        {"$match": {"user_id": user_id}},
        {"$limit": top_k},
        {
            "$project": {
                "_id": 0,
                "chunk_text": 1,
                "page": 1,
                "pdf_id": 1,
                "pdf_name": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    return list(col.aggregate(pipeline))


# ==========================
# Hinglish Detection ✅
# ==========================
def detect_language_style(text: str) -> str:
    """
    Returns 'hinglish' or 'english'
    Hinglish = romanized Hindi words written in English letters.
    """
    if not text:
        return "english"

    t = text.lower().strip()

    # common roman hindi words
    hinglish_words = [
        "apna", "bhai", "yaar", "hai", "haan", "han", "nahi", "nhi", "kya", "kyu", "kyun",
        "kaise", "mera", "meri", "teri", "tum", "aap", "hum", "ham", "matlab",
        "samajh", "samjha", "bata", "batao", "dikha", "dikhao", "karo", "karna",
        "chahiye", "krna", "krega", "krdo", "pls", "please", "sir", "maam",
        "college", "nsut", "dtu", "iit"
    ]

    hits = 0
    for w in hinglish_words:
        if f" {w} " in f" {t} ":
            hits += 1

    return "hinglish" if hits >= 2 else "english"


def ollama_answer(question: str, contexts):
    context_text = ""
    for c in contexts:
        context_text += f"\n[Page {c.get('page')}] {c.get('chunk_text')}\n"

    lang = detect_language_style(question)

    if lang == "hinglish":
        language_instruction = """
Reply strictly in Hinglish (Hindi written in English letters).
Example words you can use: "haan", "nahi", "aap", "tum", "samajh", "kya", etc.
Do NOT use Devanagari Hindi. Use Hinglish only.
"""
    else:
        language_instruction = """
Reply strictly in English.
"""

    prompt = f"""
You are LearnLens AI Tutor.

RULES:
1) Answer ONLY using the Notes Context.
2) If the answer is not present, reply exactly:
   "I couldn't find this in your notes."
3) Include citations like (Page X).
4) Language rule:
{language_instruction}

Notes Context:
{context_text}

User Question:
{question}
"""

    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp["message"]["content"]


def generate_summary(col, user_id, pdf_id: str):
    docs = list(col.find(
        {"pdf_id": pdf_id, "user_id": user_id},
        {"chunk_text": 1, "page": 1, "_id": 0}
    ).limit(40))

    combined = "\n".join([f"[Page {d['page']}] {d['chunk_text']}" for d in docs])

    prompt = f"""
You are a study assistant. Create a well-structured summary from these notes.

NOTES:
{combined}

Output Format:
- Title
- Key Concepts (bullets)
- Important definitions/formulas (if any)
- 5 short revision points
"""

    resp = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"]


# ==========================
# QUIZ HELPERS
# ==========================
def extract_json_array(raw: str):
    if not raw:
        return None

    raw = raw.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    start = raw.find("[")
    end = raw.rfind("]")

    if start == -1 or end == -1 or end <= start:
        return None

    return raw[start:end + 1]


def difficulty_rules(difficulty: str) -> str:
    difficulty = difficulty.strip().lower()

    if difficulty == "easy":
        return """
DIFFICULTY: EASY
- Basic concept questions
- Direct formula/definition based
- No tricky options
- Clear correct answer
"""
    if difficulty == "hard":
        return """
DIFFICULTY: HARD
- Tricky conceptual questions
- Options should be close/confusing
- Include trap options
- Deeper understanding required
"""
    return """
DIFFICULTY: MEDIUM
- Mix of factual + conceptual questions
- Some tricky options
- Not too lengthy
"""


def generate_quiz_notes_only(col, user_id, pdf_id: str, difficulty: str):
    docs = list(col.find(
        {"pdf_id": pdf_id, "user_id": user_id},
        {"chunk_text": 1, "_id": 0}
    ).limit(35))

    context = "\n".join([d["chunk_text"] for d in docs])
    diff = difficulty_rules(difficulty)

    prompt = f"""
You are a quiz generator.

Generate EXACTLY 10 MCQs from the notes below.

{diff}

Return STRICT VALID JSON ONLY (no markdown, no extra text).
Rules:
- Must return a JSON array of exactly 10 objects
- Use double quotes only
- No trailing commas
- Options must have 4 choices
- Answer must be exactly one: "A", "B", "C", "D"

Schema:
[
  {{
    "question": "....",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "answer": "A",
    "explanation": "...."
  }}
]

NOTES:
{context}
"""

    resp = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
    raw = resp["message"]["content"]

    extracted = extract_json_array(raw)
    if extracted is None:
        return raw, None

    try:
        return raw, json.loads(extracted)
    except Exception:
        try:
            repaired = re.sub(r",\s*([\]}])", r"\1", extracted)
            return raw, json.loads(repaired)
        except Exception:
            return raw, None


def generate_quiz_notes_plus_pyq(col, user_id, pdf_id: str, pyq_text: str, difficulty: str):
    docs = list(col.find(
        {"pdf_id": pdf_id, "user_id": user_id},
        {"chunk_text": 1, "page": 1, "_id": 0}
    ).limit(45))

    notes_context = "\n".join([f"[Page {d['page']}] {d['chunk_text']}" for d in docs])
    pyq_text = pyq_text[:12000]
    diff = difficulty_rules(difficulty)

    prompt = f"""
You are an exam-style MCQ generator.

TASK:
1) Analyze the PYQ paper and understand its MCQ pattern.
2) Generate EXACTLY 10 MCQs from NOTES in the SAME pattern.

{diff}

IMPORTANT:
- Generate ONLY MCQs
- Return STRICT VALID JSON ONLY
- No markdown, no extra text

Rules:
- Must return JSON array of exactly 10 objects
- Use double quotes only
- No trailing commas
- Options must have 4 choices
- Answer must be exactly one: "A", "B", "C", "D"

Schema:
[
  {{
    "question": "....",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "answer": "A",
    "explanation": "...."
  }}
]

PYQ PAPER:
{pyq_text}

NOTES:
{notes_context}
"""

    resp = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
    raw = resp["message"]["content"]

    extracted = extract_json_array(raw)
    if extracted is None:
        return raw, None

    try:
        return raw, json.loads(extracted)
    except Exception:
        try:
            repaired = re.sub(r",\s*([\]}])", r"\1", extracted)
            return raw, json.loads(repaired)
        except Exception:
            return raw, None


def option_starts_with(opt: str, letter: str) -> bool:
    return opt.strip().upper().startswith(letter.strip().upper())


def find_option_text(options, letter: str):
    for opt in options:
        if option_starts_with(opt, letter):
            return opt
    return None


# ==========================
# UI
# ==========================
st.title("📘 LearnLens — Exam Intelligence Platform")

col = get_mongo_collection()
embed_model = get_embedding_model()
user_id = "test_user"

tab1, tab2, tab3 = st.tabs(["📤 Upload Notes", "💬 Ask Questions", "🧠 Summary + Quiz"])


# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("📤 Upload PDF Notes")
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_pdf:
        pdf_id, filename, pdf_path = save_uploaded_pdf(uploaded_pdf)
        st.success(f"PDF saved ✅ — {filename}")
        st.write("PDF ID:", pdf_id)

        pages = extract_pages_from_pdf(pdf_path)
        st.info(f"Extracted text from {len(pages)} pages (non-empty).")

        if st.button("🚀 Ingest PDF (Chunk + Embed + Store in MongoDB)"):
            with st.spinner("🗑️ Deleting previous PDFs from MongoDB..."):
                deleted_count = delete_previous_user_pdfs(col, user_id)
            st.info(f"Deleted {deleted_count} old chunks for user: {user_id}")

            with st.spinner("Chunking + embedding + inserting into MongoDB..."):
                inserted = store_chunks_in_mongo(pdf_id, filename, user_id, pages, col, embed_model)

            st.success(f"✅ Ingestion complete: inserted {inserted} chunks into MongoDB.")


# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("💬 Chat with your Notes (RAG)")

    question = st.text_input("Ask a question based on your uploaded notes (English or Hinglish):")

    if st.button("🔎 Answer using Notes"):
        if not question.strip():
            st.warning("Please type a question.")
        else:
            with st.spinner("Searching notes using Vector Search..."):
                contexts = vector_search(question, col, embed_model, user_id=user_id, top_k=4)

            if not contexts:
                st.error("No matching content found in DB.")
            else:
                with st.spinner("Generating answer using Ollama..."):
                    ans = ollama_answer(question, contexts)

                st.markdown("## ✅ Answer")
                st.write(ans)


# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("🧠 Summary + Quiz")

    pdf_ids = col.distinct("pdf_id", {"user_id": user_id})
    if not pdf_ids:
        st.warning("No PDFs ingested yet. Upload and ingest a PDF first.")
    else:
        selected_pdf = st.selectbox("Select PDF ID", pdf_ids)

        c1, c2 = st.columns(2)

        with c1:
            if st.button("📝 Generate Summary"):
                with st.spinner("Generating summary using Ollama..."):
                    summary = generate_summary(col, user_id, selected_pdf)
                st.markdown("### ✅ Summary")
                st.write(summary)

        with c2:
            if st.button("❓ Generate Quiz (10 MCQs)"):
                st.session_state.show_quiz_mode = True
                st.session_state.quiz_mode = None

        # difficulty selector
        st.session_state.quiz_difficulty = st.selectbox(
            "🎚 Select Difficulty Level",
            ["Easy", "Medium", "Hard"],
            index=["Easy", "Medium", "Hard"].index(st.session_state.quiz_difficulty)
        )

        # mode selector
        if st.session_state.show_quiz_mode:
            st.markdown("### ✅ Select Quiz Mode")
            b1, b2 = st.columns(2)

            with b1:
                if st.button("🎲 MCQs from Notes Only"):
                    st.session_state.quiz_mode = "notes"

            with b2:
                if st.button("📄 MCQs from Notes + PYQs Pattern"):
                    st.session_state.quiz_mode = "notes_pyq"

        # notes only
        if st.session_state.quiz_mode == "notes":
            with st.spinner("Generating quiz..."):
                raw, quiz = generate_quiz_notes_only(
                    col, user_id, selected_pdf, st.session_state.quiz_difficulty
                )

            if quiz is None:
                st.error("Quiz JSON parsing failed. Raw output shown below:")
                st.code(raw)
            else:
                st.session_state.quiz_data = quiz
                st.session_state.quiz_checked = {}
                st.session_state.quiz_result = {}
                st.session_state.quiz_score = 0
                st.success(f"✅ Quiz generated (Difficulty: {st.session_state.quiz_difficulty})!")

        # notes + pyq
        if st.session_state.quiz_mode == "notes_pyq":
            st.markdown("### 📄 Upload PYQ PDF")
            pyq_pdf = st.file_uploader("Upload PYQ PDF", type=["pdf"], key="pyq_uploader")

            if pyq_pdf:
                _, _, pyq_path = save_uploaded_pdf(pyq_pdf)
                pyq_pages = extract_pages_from_pdf(pyq_path)
                pyq_text = "\n".join([p["text"] for p in pyq_pages])

                if st.button("✅ Generate MCQs using Notes + PYQ Pattern"):
                    with st.spinner("Analyzing PYQ pattern and generating quiz..."):
                        raw, quiz = generate_quiz_notes_plus_pyq(
                            col, user_id, selected_pdf, pyq_text, st.session_state.quiz_difficulty
                        )

                    if quiz is None:
                        st.error("Quiz JSON parsing failed. Raw output shown below:")
                        st.code(raw)
                    else:
                        st.session_state.quiz_data = quiz
                        st.session_state.quiz_checked = {}
                        st.session_state.quiz_result = {}
                        st.session_state.quiz_score = 0
                        st.success(f"✅ Quiz generated (PYQ Mode, Difficulty: {st.session_state.quiz_difficulty})!")

        # display quiz
        quiz = st.session_state.quiz_data
        if quiz is not None:
            st.markdown(f"### 🏆 Score: **{st.session_state.quiz_score} / {len(quiz)}**")
            st.divider()

            for i, q in enumerate(quiz):
                checked = st.session_state.quiz_checked.get(i, False)

                border = "#d1d5db"
                bg = "#ffffff"

                if checked:
                    if st.session_state.quiz_result[i]["is_correct"]:
                        border = "#16a34a"
                        bg = "#f0fdf4"
                    else:
                        border = "#dc2626"
                        bg = "#fef2f2"

                st.markdown(
                    f"""
                    <div style="
                        border: 3px solid {border};
                        background-color: {bg};
                        padding: 18px;
                        border-radius: 16px;
                        margin-bottom: 12px;
                    ">
                        <h3 style="margin:0;">Q{i+1}. {q['question']}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                choice = st.radio("Choose:", q["options"], key=f"quiz_choice_{i}")

                if st.button(f"✅ Check Q{i+1}", key=f"check_btn_{i}"):
                    if not checked:
                        correct_letter = q["answer"].strip().upper()
                        is_correct = option_starts_with(choice, correct_letter)

                        st.session_state.quiz_checked[i] = True
                        st.session_state.quiz_result[i] = {
                            "is_correct": is_correct,
                            "correct_letter": correct_letter
                        }

                        if is_correct:
                            st.session_state.quiz_score += 1

                        st.rerun()

                if checked:
                    is_correct = st.session_state.quiz_result[i]["is_correct"]
                    correct_letter = st.session_state.quiz_result[i]["correct_letter"]

                    if is_correct:
                        st.success("✅ Correct!")
                    else:
                        correct_option = find_option_text(q["options"], correct_letter)
                        st.error("❌ Wrong!")
                        st.info(f"✅ Correct Answer: **{correct_option}**")
                        st.warning(f"📌 Explanation: {q.get('explanation', 'No explanation provided')}")

                st.divider()

            total_checked = sum(1 for x in st.session_state.quiz_checked.values() if x)

            if total_checked == len(quiz):
                st.balloons()
                st.success(f"🎉 Quiz Completed! Final Score: **{st.session_state.quiz_score} / {len(quiz)}**")
            else:
                st.info(f"Progress: {total_checked}/{len(quiz)} questions checked.")