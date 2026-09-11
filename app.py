"""
AI-Powered Laboratory Diagnostics & Result Interpretation System
------------------------------------------------------------------
- Upload a lab report PDF (or enter values manually from 100+ standard tests)
- PDF text is extracted (PyMuPDF), chunked, embedded (sentence-transformers)
  and stored in a local FAISS vector index (fully open-source, no paid DB)
- Relevant chunks + patient context (age/sex/fasting/history) are sent
  along with the question to a free/open-weight model hosted on Groq

IMPORTANT: This tool is for educational/informational purposes only.
It does NOT provide medical diagnosis and is not a substitute for
professional medical advice. Always consult a qualified clinician.

Reference ranges are compiled from widely-published, standard adult
clinical reference intervals (the kind printed by most hospital labs).
Exact cutoffs vary by laboratory, analyzer, and method — always defer
to the reference range printed on the actual report.
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
# REFERENCE RANGES — 100+ tests across standard clinical categories
# General adult values; many labs show slightly different cutoffs
# depending on method/analyzer, so treat these as a starting reference.
# ---------------------------------------------------------------------
REFERENCE_RANGES = {
    "Hematology": {
        "Hemoglobin":            {"unit": "g/dL",      "male": (13.5, 17.5), "female": (12.0, 15.5)},
        "Hematocrit":            {"unit": "%",         "male": (38.8, 50.0), "female": (34.9, 44.5)},
        "RBC Count":             {"unit": "x10^6/uL",  "male": (4.7, 6.1),   "female": (4.2, 5.4)},
        "WBC Count":             {"unit": "x10^9/L",   "range": (4.0, 11.0)},
        "Platelet Count":        {"unit": "x10^9/L",   "range": (150, 450)},
        "MCV":                   {"unit": "fL",        "range": (80, 100)},
        "MCH":                   {"unit": "pg",        "range": (27, 33)},
        "MCHC":                  {"unit": "g/dL",      "range": (32, 36)},
        "RDW":                   {"unit": "%",         "range": (11.5, 14.5)},
        "Neutrophils":           {"unit": "%",         "range": (40, 60)},
        "Lymphocytes":           {"unit": "%",         "range": (20, 40)},
        "Monocytes":             {"unit": "%",         "range": (2, 8)},
        "Eosinophils":           {"unit": "%",         "range": (1, 4)},
        "Basophils":             {"unit": "%",         "range": (0.5, 1.0)},
        "ESR":                   {"unit": "mm/hr",     "male": (0, 15),      "female": (0, 20)},
        "Reticulocyte Count":    {"unit": "%",         "range": (0.5, 2.5)},
    },
    "Coagulation": {
        "Prothrombin Time (PT)": {"unit": "sec",  "range": (11, 13.5)},
        "INR":                   {"unit": "",     "range": (0.8, 1.1)},
        "aPTT":                  {"unit": "sec",  "range": (25, 35)},
        "Fibrinogen":            {"unit": "mg/dL","range": (200, 400)},
        "D-Dimer":               {"unit": "ng/mL FEU", "range": (0, 500)},
    },
    "Electrolytes & Metabolic": {
        "Sodium":                {"unit": "mmol/L", "range": (135, 145)},
        "Potassium":             {"unit": "mmol/L", "range": (3.5, 5.1)},
        "Chloride":              {"unit": "mmol/L", "range": (98, 107)},
        "Bicarbonate (CO2)":     {"unit": "mmol/L", "range": (22, 29)},
        "Calcium (Total)":       {"unit": "mg/dL",  "range": (8.5, 10.5)},
        "Ionized Calcium":       {"unit": "mmol/L", "range": (1.1, 1.3)},
        "Magnesium":             {"unit": "mg/dL",  "range": (1.7, 2.2)},
        "Phosphate":             {"unit": "mg/dL",  "range": (2.5, 4.5)},
        "Anion Gap":             {"unit": "mmol/L", "range": (8, 16)},
        "Osmolality":            {"unit": "mOsm/kg","range": (275, 295)},
    },
    "Renal": {
        "Blood Urea Nitrogen (BUN)":       {"unit": "mg/dL",         "range": (7, 20)},
        "Creatinine":                      {"unit": "mg/dL",         "male": (0.7, 1.3), "female": (0.6, 1.1)},
        "eGFR":                            {"unit": "mL/min/1.73m2", "range": (90, 120)},
        "Uric Acid":                       {"unit": "mg/dL",         "male": (3.4, 7.0), "female": (2.4, 6.0)},
        "Cystatin C":                      {"unit": "mg/L",          "range": (0.5, 1.0)},
        "Urine Albumin-Creatinine Ratio":  {"unit": "mg/g",          "range": (0, 30)},
    },
    "Liver Function": {
        "ALT (SGPT)":            {"unit": "U/L",   "range": (7, 56)},
        "AST (SGOT)":            {"unit": "U/L",   "range": (10, 40)},
        "ALP":                   {"unit": "U/L",   "range": (44, 147)},
        "GGT":                   {"unit": "U/L",   "male": (8, 61), "female": (5, 36)},
        "Total Bilirubin":       {"unit": "mg/dL", "range": (0.1, 1.2)},
        "Direct Bilirubin":      {"unit": "mg/dL", "range": (0.0, 0.3)},
        "Indirect Bilirubin":    {"unit": "mg/dL", "range": (0.2, 0.9)},
        "Albumin":               {"unit": "g/dL",  "range": (3.5, 5.0)},
        "Total Protein":         {"unit": "g/dL",  "range": (6.0, 8.3)},
    },
    "Lipid Panel": {
        "Total Cholesterol":     {"unit": "mg/dL", "range": (0, 200)},
        "LDL Cholesterol":       {"unit": "mg/dL", "range": (0, 100)},
        "HDL Cholesterol":       {"unit": "mg/dL", "range": (40, 60)},
        "Triglycerides":         {"unit": "mg/dL", "range": (0, 150)},
        "VLDL Cholesterol":      {"unit": "mg/dL", "range": (5, 40)},
        "Non-HDL Cholesterol":   {"unit": "mg/dL", "range": (0, 130)},
    },
    "Glucose & Diabetes": {
        "Fasting Glucose":               {"unit": "mg/dL",  "range": (70, 99)},
        "Random Glucose":                {"unit": "mg/dL",  "range": (70, 140)},
        "Postprandial Glucose (2-hr)":   {"unit": "mg/dL",  "range": (70, 140)},
        "HbA1c":                         {"unit": "%",      "range": (4.0, 5.6)},
        "Fasting Insulin":               {"unit": "uIU/mL", "range": (2.6, 24.9)},
        "C-Peptide":                     {"unit": "ng/mL",  "range": (0.8, 3.1)},
    },
    "Thyroid": {
        "TSH":                   {"unit": "mIU/L",  "range": (0.4, 4.0)},
        "Free T4":               {"unit": "ng/dL",  "range": (0.8, 1.8)},
        "Free T3":               {"unit": "pg/mL",  "range": (2.3, 4.2)},
        "Total T4":              {"unit": "ug/dL",  "range": (5.0, 12.0)},
        "Total T3":              {"unit": "ng/dL",  "range": (80, 200)},
        "Anti-TPO Antibodies":   {"unit": "IU/mL",  "range": (0, 34)},
    },
    "Cardiac Markers": {
        "Troponin I":            {"unit": "ng/mL", "range": (0, 0.04)},
        "Troponin T":            {"unit": "ng/mL", "range": (0, 0.01)},
        "CK-MB":                 {"unit": "ng/mL", "range": (0, 5)},
        "BNP":                   {"unit": "pg/mL", "range": (0, 100)},
        "NT-proBNP":             {"unit": "pg/mL", "range": (0, 125)},
        "CRP":                   {"unit": "mg/L",  "range": (0, 10)},
        "hs-CRP":                {"unit": "mg/L",  "range": (0, 3)},
    },
    "Iron Studies": {
        "Serum Iron":                {"unit": "ug/dL", "male": (65, 175), "female": (50, 170)},
        "Ferritin":                  {"unit": "ng/mL", "male": (24, 336), "female": (11, 307)},
        "TIBC":                      {"unit": "ug/dL", "range": (250, 450)},
        "Transferrin Saturation":    {"unit": "%",     "range": (20, 50)},
        "Transferrin":               {"unit": "mg/dL", "range": (200, 360)},
    },
    "Vitamins & Minerals": {
        "Vitamin D (25-OH)":     {"unit": "ng/mL", "range": (30, 100)},
        "Vitamin B12":           {"unit": "pg/mL", "range": (200, 900)},
        "Folate":                {"unit": "ng/mL", "range": (2.7, 17.0)},
        "Zinc":                  {"unit": "ug/dL", "range": (60, 120)},
    },
    "Hormones": {
        "Testosterone (Total)":  {"unit": "ng/dL", "male": (280, 1100), "female": (15, 70)},
        "Estradiol":             {"unit": "pg/mL", "range": (15, 350)},
        "Progesterone":          {"unit": "ng/mL", "range": (0.1, 25)},
        "Cortisol (AM)":         {"unit": "ug/dL", "range": (6, 23)},
        "Prolactin":             {"unit": "ng/mL", "male": (4, 15.2), "female": (4.8, 23.3)},
        "LH":                    {"unit": "mIU/mL","range": (1.7, 8.6)},
        "FSH":                   {"unit": "mIU/mL","range": (1.5, 12.4)},
        "DHEA-S":                {"unit": "ug/dL", "male": (80, 560), "female": (35, 430)},
    },
    "Tumor Markers": {
        "PSA (Total)":           {"unit": "ng/mL", "range": (0, 4.0)},
        "CA-125":                {"unit": "U/mL",  "range": (0, 35)},
        "CA 19-9":               {"unit": "U/mL",  "range": (0, 37)},
        "AFP":                   {"unit": "ng/mL", "range": (0, 10)},
        "CEA":                   {"unit": "ng/mL", "range": (0, 3.0)},
    },
    "Urinalysis": {
        "Urine Specific Gravity":  {"unit": "",        "range": (1.005, 1.030)},
        "Urine pH":                {"unit": "",        "range": (4.5, 8.0)},
        "Urine Protein (24h)":     {"unit": "mg/24h",  "range": (0, 150)},
        "Urine Microalbumin":      {"unit": "mg/L",    "range": (0, 20)},
    },
    "Other Enzymes & Panels": {
        "Amylase":       {"unit": "U/L",    "range": (28, 100)},
        "Lipase":        {"unit": "U/L",    "range": (10, 140)},
        "LDH":           {"unit": "U/L",    "range": (140, 280)},
        "Ammonia":       {"unit": "umol/L", "range": (15, 45)},
        "Homocysteine":  {"unit": "umol/L", "range": (5, 15)},
    },
}

# Flat lookup: test name -> info (with its category attached)
FLAT_TESTS = {}
for _cat, _tests in REFERENCE_RANGES.items():
    for _name, _info in _tests.items():
        FLAT_TESTS[_name] = {**_info, "category": _cat}

TOTAL_TEST_COUNT = len(FLAT_TESTS)

# ---------------------------------------------------------------------
# CACHED RESOURCES
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    # Free, open-source, runs locally — no API key needed
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_groq_client(api_key: str):
    return Groq(api_key=api_key)


# Known-good free/open-weight chat models on Groq as of Sept 2026.
# Groq periodically deprecates models — see console.groq.com/docs/models.
# The app also tries to fetch the live list below (best-effort).
FALLBACK_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "groq/compound-mini",
]

EXCLUDE_SUBSTRINGS = ["whisper", "tts", "guard", "prompt-guard", "orpheus"]


def fetch_live_models(api_key: str):
    """Best-effort fetch of currently active chat-capable Groq models."""
    try:
        client = get_groq_client(api_key)
        models = client.models.list()
        ids = [m.id for m in models.data]
        chat_models = [m for m in ids if not any(x in m.lower() for x in EXCLUDE_SUBSTRINGS)]
        return sorted(chat_models) if chat_models else None
    except Exception:
        return None


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
    "You are given optional patient context (age, sex, fasting status, "
    "known conditions, medications) — use it to make your explanation "
    "more relevant (e.g. age-related reference ranges, effect of fasting "
    "on glucose/lipids, interactions with a stated condition), but never "
    "state a definitive diagnosis. "
    "You must: (1) explain what each test measures in simple terms, "
    "(2) state whether values are within, above, or below the given "
    "reference range, (3) describe general, well-established reasons "
    "values in that direction are commonly seen, factoring in the "
    "patient context if provided, and (4) clearly remind the user this "
    "is general information, not a diagnosis, and that they should "
    "discuss results with a licensed doctor. "
    "Never recommend specific drug dosages or treatment plans."
)


def ask_groq(client, model, context, question, patient_context=""):
    parts = []
    if patient_context.strip():
        parts.append(f"PATIENT CONTEXT:\n{patient_context}")
    parts.append(f"LAB REPORT CONTEXT:\n{context}")
    parts.append(f"USER QUESTION:\n{question}")
    parts.append("Answer clearly, using short sections and bullet points where useful.")
    user_prompt = "\n\n".join(parts)

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

if "available_models" not in st.session_state:
    st.session_state.available_models = FALLBACK_MODELS

if st.sidebar.button("🔄 Refresh model list from Groq"):
    if groq_api_key:
        live = fetch_live_models(groq_api_key)
        if live:
            st.session_state.available_models = live
            st.sidebar.success(f"Loaded {len(live)} live models.")
        else:
            st.sidebar.warning("Could not fetch live models — using fallback list.")
    else:
        st.sidebar.warning("Enter your API key first.")

model_choice = st.sidebar.selectbox("Model", st.session_state.available_models, index=0)
st.sidebar.caption(
    "If a model errors with 'does not exist', click Refresh above, or check "
    "console.groq.com/docs/models — Groq periodically retires free-tier models."
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"📚 {TOTAL_TEST_COUNT} lab tests available across "
    f"{len(REFERENCE_RANGES)} categories. Embeddings run locally — only "
    "retrieved text + your question are sent to Groq."
)

# ---------------------------------------------------------------------
# PATIENT CONTEXT — shared by both tabs, improves interpretation quality
# ---------------------------------------------------------------------
with st.expander("🧍 Patient Context (optional, but recommended for a more relevant explanation)", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=0, step=1)
    with c2:
        sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
    with c3:
        fasting = st.radio("Fasting for this test?", ["Unknown", "Yes", "No"], horizontal=True)

    conditions = st.text_input(
        "Known conditions / symptoms (optional)",
        placeholder="e.g. type 2 diabetes, recent fever, family history of thyroid disease"
    )
    medications = st.text_input(
        "Current medications / supplements (optional)",
        placeholder="e.g. metformin, levothyroxine, iron supplement"
    )

def build_patient_context_str():
    lines = []
    if age and age > 0:
        lines.append(f"Age: {age} years")
    lines.append(f"Sex: {sex}")
    if fasting != "Unknown":
        lines.append(f"Fasting for this test: {fasting}")
    if conditions.strip():
        lines.append(f"Known conditions/symptoms: {conditions.strip()}")
    if medications.strip():
        lines.append(f"Current medications/supplements: {medications.strip()}")
    return "\n".join(lines)

patient_context_str = build_patient_context_str()

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
tab1, tab2 = st.tabs(["📄 Upload PDF Report", "✍️ Manual Entry (100+ tests)"])

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
                        answer = ask_groq(client, model_choice, context, question, patient_context_str)
                        st.markdown("### 🧾 Interpretation")
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"Groq API error: {e}")

# ---------------------- TAB 2: MANUAL ENTRY --------------------------
with tab2:
    st.subheader(f"Enter test values manually ({TOTAL_TEST_COUNT}+ tests available)")

    selected_categories = st.multiselect(
        "Choose one or more categories",
        options=list(REFERENCE_RANGES.keys()),
        default=[],
    )

    selected_tests = []
    for cat in selected_categories:
        with st.expander(f"{cat} ({len(REFERENCE_RANGES[cat])} tests)", expanded=True):
            picked = st.multiselect(
                f"Select tests from {cat}",
                options=list(REFERENCE_RANGES[cat].keys()),
                key=f"pick_{cat}",
            )
            selected_tests.extend(picked)

    entries = []
    if selected_tests:
        st.markdown("#### Enter values")
        cols = st.columns(2)
        for i, test in enumerate(selected_tests):
            info = FLAT_TESTS[test]
            unit = info["unit"]
            label = f"{test} ({unit})" if unit else test
            with cols[i % 2]:
                value = st.number_input(label, min_value=0.0, format="%.3f", key=f"val_{test}")
                entries.append((test, value, unit))

    if entries and st.button("Compute Status Table"):
        rows = []
        for test, value, unit in entries:
            info = FLAT_TESTS[test]
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
            unit_str = f" {unit}" if unit else ""
            rows.append(f"- **{test}** ({info['category']}): {value}{unit_str} (reference {low}-{high}{unit_str}) → {status}")

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
                        answer = ask_groq(
                            client, model_choice, st.session_state.manual_summary,
                            m_question, patient_context_str
                        )
                        st.markdown("### 🧾 Interpretation")
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"Groq API error: {e}")

st.markdown("---")
st.caption(
    "Built with Streamlit, PyMuPDF, Sentence-Transformers, FAISS, and Groq. "
    "Reference ranges are general adult values compiled from widely-published "
    "clinical references and can vary by laboratory, analyzer, and method — "
    "always check the range printed on your actual report."
)
