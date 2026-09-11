"""
AI-Powered Laboratory Diagnostics & Result Interpretation System
------------------------------------------------------------------
- Upload a lab report PDF (or enter values manually)
- PDF text is extracted (PyMuPDF), chunked, embedded (sentence-transformers)
  and stored in a local FAISS vector index (fully open-source, no paid DB)
- Relevant chunks are retrieved for a question and sent, along with the
  question, to a free/open-weight model hosted on Groq for interpretation

IMPORTANT: This tool is for educational/informational purposes only.
It does NOT provide medical diagnosis and is not a substitute for
professional medical advice. Always consult a qualified clinician.
"""

import os
import re
import numpy as np
import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq

# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="AI Lab Report Interpreter",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 AI-Powered Lab Diagnostics & Result Interpreter")
st.warning(
    "⚠️ **Disclaimer:** This app provides general, educational information "
    "only. It is NOT a medical diagnosis and does NOT replace advice from "
    "a licensed doctor or laboratory professional. Always confirm results "
    "and next steps with a qualified clinician."
)

# ---------------------------------------------------------------------
# REFERENCE RANGES (general adult ranges; individual labs may vary
# slightly — always confirm against the reference range printed on the
# actual lab report, since methods/units differ by laboratory).
# ---------------------------------------------------------------------
REFERENCE_RANGES = {
    "Hemoglobin (Hb)":       {"unit": "g/dL",     "male": (13.5, 17.5), "female": (12.0, 15.5)},
    "WBC Count":             {"unit": "x10^9/L",  "range": (4.0, 11.0)},
    "Platelet Count":        {"unit": "x10^9/L",  "range": (150, 450)},
    "Fasting Blood Glucose": {"unit": "mg/dL",    "range": (70, 99)},
    "HbA1c":                 {"unit": "%",        "range": (4.0, 5.6)},
    "Total Cholesterol":     {"unit": "mg/dL",    "range": (0, 200)},
    "LDL Cholesterol":       {"unit": "mg/dL",    "range": (0, 100)},
    "HDL Cholesterol":       {"unit": "mg/dL",    "range": (40, 60)},
    "Triglycerides":         {"unit": "mg/dL",    "range": (0, 150)},
    "ALT (SGPT)":            {"unit": "U/L",      "range": (7, 56)},
    "AST (SGOT)":            {"unit": "U/L",      "range": (10, 40)},
    "Creatinine":            {"unit": "mg/dL",    "male": (0.7, 1.3), "female": (0.6, 1.1)},
    "Blood Urea":            {"unit": "mg/dL",    "range": (7, 20)},
    "TSH":                   {"unit": "mIU/L",    "range": (0.4, 4.0)},
    "Sodium":                {"unit": "mmol/L",   "range": (135, 145)},
    "Potassium":             {"unit": "mmol/L",   "range": (3.5, 5.1)},
}

# ---------------------------------------------------------------------
# CACHED RESOURCES
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    # Free, open-source, runs locally — no API key needed
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_groq_client(api_key: str):
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------
# PDF -> TEXT -> CHUNKS -> EMBEDDINGS -> FAISS INDEX
# ---------------------------------------------------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int = 220, overlap: int = 40):
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def build_faiss_index(chunks, embedder):
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve_relevant_chunks(query, index, chunks, embedder, k=4):
    if index is None or not chunks:
        return []
    q_emb = embedder.encode([query]).astype("float32")
    k = min(k, len(chunks))
    _, indices = index.search(q_emb, k)
    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]


# ---------------------------------------------------------------------
# GROQ CALL
# ---------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a careful, plain-language medical information assistant. "
    "You help a layperson understand their lab report. "
    "You must: (1) explain what each test measures in simple terms, "
    "(2) state whether values are within, above, or below the given "
    "reference range, (3) describe general, well-established reasons "
    "values in that direction are commonly seen, and (4) clearly remind "
    "the user this is general information, not a diagnosis, and that "
    "they should discuss results with a licensed doctor. "
    "Never state a definitive diagnosis. Never recommend specific drug "
    "dosages or treatment plans."
)


def ask_groq(client, model, context, question):
    user_prompt = (
        f"LAB REPORT CONTEXT:\n{context}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        "Answer clearly, using short sections and bullet points where useful."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1200,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------
# SIDEBAR — API KEY & MODEL
# ---------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

default_key = os.getenv("GROQ_API_KEY", "")
try:
    default_key = st.secrets.get("GROQ_API_KEY", default_key)
except Exception:
    pass

groq_api_key = st.sidebar.text_input(
    "Groq API Key", type="password", value=default_key,
    help="Get a free key at https://console.groq.com/keys"
)

# Free/open-weight models currently served on Groq.
# Check https://console.groq.com/docs/models for the latest list —
# Groq periodically adds/retires models.
model_choice = st.sidebar.selectbox(
    "Model",
    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "This app runs embeddings locally (sentence-transformers) and only "
    "sends retrieved text + your question to Groq for interpretation."
)

# ---------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None
if "manual_summary" not in st.session_state:
    st.session_state.manual_summary = ""

embedder = load_embedder()

# ---------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------
tab1, tab2 = st.tabs(["📄 Upload PDF Report", "✍️ Manual Entry"])

# ------------------------- TAB 1: PDF -------------------------------
with tab1:
    st.subheader("Upload your lab report (PDF)")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Process PDF"):
            with st.spinner("Extracting text, chunking, and building index..."):
                raw_text = extract_text_from_pdf(uploaded_file)
                chunks = chunk_text(raw_text)
                if not chunks:
                    st.error("Could not extract any text from this PDF. It may be a scanned image — try manual entry instead.")
                else:
                    index = build_faiss_index(chunks, embedder)
                    st.session_state.chunks = chunks
                    st.session_state.index = index
                    st.success(f"Processed! Created {len(chunks)} text chunks from the report.")

    if st.session_state.chunks:
        st.markdown("---")
        question = st.text_area(
            "Ask a question about this report",
            value="Please explain these lab results in simple terms and flag any values that look abnormal.",
            height=80,
        )
        if st.button("🔍 Analyze Report", type="primary"):
            if not groq_api_key:
                st.error("Please enter your Groq API key in the sidebar first.")
            else:
                with st.spinner("Retrieving relevant sections and asking the model..."):
                    relevant = retrieve_relevant_chunks(
                        question, st.session_state.index, st.session_state.chunks, embedder, k=4
                    )
                    context = "\n---\n".join(relevant)
                    client = get_groq_client(groq_api_key)
                    try:
                        answer = ask_groq(client, model_choice, context, question)
                        st.markdown("### 🧾 Interpretation")
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"Groq API error: {e}")

# ---------------------- TAB 2: MANUAL ENTRY --------------------------
with tab2:
    st.subheader("Enter test values manually")
    sex = st.radio("Sex (for tests with sex-specific ranges)", ["Male", "Female"], horizontal=True)

    selected_tests = st.multiselect(
        "Select the tests you want to enter",
        options=list(REFERENCE_RANGES.keys()),
    )

    entries = []
    if selected_tests:
        st.markdown("#### Enter values")
        cols = st.columns(2)
        for i, test in enumerate(selected_tests):
            info = REFERENCE_RANGES[test]
            unit = info["unit"]
            with cols[i % 2]:
                value = st.number_input(f"{test} ({unit})", min_value=0.0, format="%.2f", key=f"val_{test}")
                entries.append((test, value, unit))

    if entries and st.button("Compute Status Table"):
        rows = []
        for test, value, unit in entries:
            info = REFERENCE_RANGES[test]
            if "male" in info:
                low, high = info["male"] if sex == "Male" else info["female"]
            else:
                low, high = info["range"]
            if value < low:
                status = "🔵 Low"
            elif value > high:
                status = "🔴 High"
            else:
                status = "🟢 Normal"
            rows.append(f"- **{test}**: {value} {unit} (reference {low}-{high} {unit}) → {status}")

        summary = "\n".join(rows)
        st.session_state.manual_summary = summary
        st.markdown("### Results")
        st.markdown(summary)

    if st.session_state.manual_summary:
        st.markdown("---")
        m_question = st.text_area(
            "Ask a question about these results",
            value="Please explain what these results mean in simple terms and what abnormal values could generally indicate.",
            height=80,
            key="manual_question",
        )
        if st.button("🔍 Interpret My Results", type="primary"):
            if not groq_api_key:
                st.error("Please enter your Groq API key in the sidebar first.")
            else:
                with st.spinner("Asking the model..."):
                    client = get_groq_client(groq_api_key)
                    try:
                        answer = ask_groq(client, model_choice, st.session_state.manual_summary, m_question)
                        st.markdown("### 🧾 Interpretation")
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"Groq API error: {e}")

st.markdown("---")
st.caption(
    "Built with Streamlit, PyMuPDF, Sentence-Transformers, FAISS, and Groq. "
    "Reference ranges are general adult values and can vary by laboratory and method."
)
