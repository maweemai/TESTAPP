"""
AI-Powered Laboratory Diagnostics & Result Interpretation System
------------------------------------------------------------------
- Upload a lab report as PDF or image (JPG/PNG) — or enter values manually
  from 100+ standard tests
- Text is extracted via PyMuPDF (text-based PDFs) with automatic OCR
  fallback (Tesseract, via pytesseract) for scanned PDFs and photos
- Patient details (name, age, sex) are auto-detected from the document
  where possible and pre-filled — always shown for the user to verify/edit
- Text is chunked, embedded locally (sentence-transformers), stored in a
  local FAISS vector index (fully open-source, no paid DB)
- Relevant chunks + patient context are sent to a free/open-weight model
  hosted on Groq for a plain-language interpretation

IMPORTANT: This tool is for educational/informational purposes only.
It does NOT provide medical diagnosis and is not a substitute for
professional medical advice. Always consult a qualified clinician.

Reference ranges are compiled from widely-published, standard adult
clinical reference intervals. Exact cutoffs vary by laboratory, analyzer,
and method — always defer to the range printed on the actual report.

SETUP NOTE (OCR): OCR requires the Tesseract binary in addition to the
`pytesseract` Python package. Locally: install it via your OS package
manager (e.g. `brew install tesseract` / `apt install tesseract-ocr` /
Windows installer). On Streamlit Community Cloud: add a `packages.txt`
file (next to requirements.txt) containing the single line
`tesseract-ocr` so the cloud environment installs the binary too.
"""

import os
import re
import io
import numpy as np
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
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
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_groq_client(api_key: str):
    return Groq(api_key=api_key)


FALLBACK_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "groq/compound-mini",
]
EXCLUDE_SUBSTRINGS = ["whisper", "tts", "guard", "prompt-guard", "orpheus"]


def fetch_live_models(api_key: str):
    try:
        client = get_groq_client(api_key)
        models = client.models.list()
        ids = [m.id for m in models.data]
        chat_models = [m for m in ids if not any(x in m.lower() for x in EXCLUDE_SUBSTRINGS)]
        return sorted(chat_models) if chat_models else None
    except Exception:
        return None


# ---------------------------------------------------------------------
# OCR HELPERS
# ---------------------------------------------------------------------
def ocr_image(image: Image.Image) -> str:
    """Run Tesseract OCR on a PIL image. Returns '' on failure with a UI warning."""
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        st.error(
            "OCR failed — Tesseract may not be installed in this environment. "
            f"Details: {e}\n\nSee the setup note at the top of app.py."
        )
        return ""


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from a PDF. Falls back to OCR page-by-page for scanned pages."""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    parts = []
    for page in doc:
        page_text = page.get_text()
        if len(page_text.strip()) < 20:  # likely a scanned/image-only page
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_text = ocr_image(img)
        parts.append(page_text)
    doc.close()
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def extract_text_from_image_file(uploaded_file) -> str:
    """Extract text from a photo/scan (JPG/PNG) via OCR."""
    image = Image.open(uploaded_file).convert("RGB")
    return re.sub(r"\s+", " ", ocr_image(image)).strip()


# ---------------------------------------------------------------------
# AUTOMATIC PATIENT INFO EXTRACTION (heuristic, always user-verifiable)
# ---------------------------------------------------------------------
def extract_patient_info(text: str) -> dict:
    """
    Best-effort extraction of Name / Age / Sex from common lab report
    layouts (including combined 'Age/Sex: 45/M' style fields used by
    many labs). This is heuristic text pattern matching, not guaranteed
    accurate — always shown to the user to verify/correct.
    """
    info = {}

    combo = re.search(
        r"age\s*/\s*sex\s*[:\-]?\s*(\d{1,3})\s*(?:y(?:rs|ears)?)?\s*/\s*(male|female|m|f)\b",
        text, re.IGNORECASE,
    )
    if combo:
        info["age"] = combo.group(1)
        info["gender"] = combo.group(2)

    if "age" not in info:
        m = re.search(r"\bage\s*[:\-]\s*(\d{1,3})", text, re.IGNORECASE)
        if m:
            info["age"] = m.group(1)

    if "gender" not in info:
        m = re.search(r"\b(?:gender|sex)\s*[:\-]\s*(male|female|m|f)\b", text, re.IGNORECASE)
        if m:
            info["gender"] = m.group(1)

    m = re.search(
        r"(?:patient\s*name|name\s*of\s*patient|name)\s*[:\-]\s*([A-Za-z.'\- ]{2,50})",
        text, re.IGNORECASE,
    )
    if m:
        candidate = m.group(1).strip()
        candidate = re.split(r"\s{2,}", candidate)[0].strip()
        if candidate:
            info["name"] = candidate

    if "gender" in info:
        g = info["gender"].strip().lower()
        info["gender"] = "Male" if g.startswith("m") else "Female"

    if "age" in info:
        try:
            info["age"] = int(info["age"])
        except ValueError:
            info.pop("age", None)

    return info


# ---------------------------------------------------------------------
# CHUNKING + FAISS
# ---------------------------------------------------------------------
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
# SIDEBAR — API KEY (hidden from visitors when set as a Streamlit secret)
# ---------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

_secret_key = None
try:
    _secret_key = st.secrets.get("GROQ_API_KEY", None)
except Exception:
    pass
_env_key = os.getenv("GROQ_API_KEY", "")
_builtin_key = _secret_key or _env_key or ""

if _builtin_key:
    groq_api_key = _builtin_key
    st.sidebar.success("✅ Using the app's built-in API key — no key needed from you.")
else:
    groq_api_key = st.sidebar.text_input(
        "Groq API Key", type="password",
        help="No built-in key found. Get a free key at https://console.groq.com/keys "
             "or add it to Streamlit Cloud's Secrets so visitors never see this field."
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
    f"📚 {TOTAL_TEST_COUNT} lab tests across {len(REFERENCE_RANGES)} categories. "
    "Embeddings and OCR run locally — only retrieved text + your question go to Groq."
)

# ---------------------------------------------------------------------
# PATIENT CONTEXT — shared by both tabs; auto-filled from uploads when
# possible, always editable. Keys are pre-initialized so uploads can
# update them via session_state before the widgets are drawn.
# ---------------------------------------------------------------------
_PC_DEFAULTS = {
    "pc_name": "", "pc_age": 0, "pc_sex": "Male",
    "pc_fasting": "Unknown", "pc_conditions": "", "pc_medications": "",
}
for _k, _v in _PC_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Apply any auto-detected patient info queued by the upload handler.
# This MUST happen before the widgets below are instantiated — Streamlit
# forbids writing to a widget's session_state key after that widget has
# already been created in the current script run.
if st.session_state.get("pending_detected"):
    _pending = st.session_state.pop("pending_detected")
    if "age" in _pending:
        st.session_state.pc_age = _pending["age"]
    if "gender" in _pending:
        st.session_state.pc_sex = _pending["gender"]
    if "name" in _pending:
        st.session_state.pc_name = _pending["name"]

with st.expander("🧍 Patient Context (auto-filled from uploads when possible — please verify)", expanded=True):
    st.text_input("Patient Name (optional, kept local only — never sent to the AI)", key="pc_name")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Age (years)", min_value=0, max_value=120, step=1, key="pc_age")
    with c2:
        st.radio("Sex", ["Male", "Female"], horizontal=True, key="pc_sex")
    with c3:
        st.radio("Fasting for this test?", ["Unknown", "Yes", "No"], horizontal=True, key="pc_fasting")
    st.text_input(
        "Known conditions / symptoms (optional)", key="pc_conditions",
        placeholder="e.g. type 2 diabetes, recent fever, family history of thyroid disease",
    )
    st.text_input(
        "Current medications / supplements (optional)", key="pc_medications",
        placeholder="e.g. metformin, levothyroxine, iron supplement",
    )


def build_patient_context_str() -> str:
    lines = []
    if st.session_state.pc_age and st.session_state.pc_age > 0:
        lines.append(f"Age: {st.session_state.pc_age} years")
    lines.append(f"Sex: {st.session_state.pc_sex}")
    if st.session_state.pc_fasting != "Unknown":
        lines.append(f"Fasting for this test: {st.session_state.pc_fasting}")
    if st.session_state.pc_conditions.strip():
        lines.append(f"Known conditions/symptoms: {st.session_state.pc_conditions.strip()}")
    if st.session_state.pc_medications.strip():
        lines.append(f"Current medications/supplements: {st.session_state.pc_medications.strip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------
for _k, _v in {"chunks": [], "index": None, "manual_summary": "", "process_message": ""}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

embedder = load_embedder()

# ---------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------
tab1, tab2 = st.tabs(["📄 Upload Report (PDF / Image)", "✍️ Manual Entry (100+ tests)"])

# ------------------------- TAB 1: UPLOAD -----------------------------
with tab1:
    st.subheader("Upload your lab report — PDF, JPG, or PNG")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "jpg", "jpeg", "png"])

    if st.session_state.process_message:
        st.info(st.session_state.process_message)

    if uploaded_file is not None:
        if st.button("Process Report"):
            with st.spinner("Extracting text (with OCR fallback if needed) and building index..."):
                suffix = uploaded_file.name.lower().split(".")[-1]
                if suffix == "pdf":
                    raw_text = extract_text_from_pdf(uploaded_file)
                else:
                    raw_text = extract_text_from_image_file(uploaded_file)

                if not raw_text or len(raw_text.strip()) < 5:
                    st.error("Could not extract any readable text from this file. Try a clearer photo/scan, or use manual entry instead.")
                else:
                    chunks = chunk_text(raw_text)
                    index = build_faiss_index(chunks, embedder)
                    st.session_state.chunks = chunks
                    st.session_state.index = index

                    detected = extract_patient_info(raw_text)
                    # Queue it — applied at the top of the script on the
                    # next run, before the Patient Context widgets exist.
                    st.session_state.pending_detected = detected

                    if detected:
                        found = ", ".join(f"{k.title()}: {v}" for k, v in detected.items())
                        st.session_state.process_message = (
                            f"Processed! Created {len(chunks)} text chunks. "
                            f"Auto-detected — {found}. Please analyze report."
                        )
                    else:
                        st.session_state.process_message = (
                            f"Processed! Created {len(chunks)} text chunks. "
                            "No patient details were auto-detected — please fill in Patient Context above manually."
                        )
                    st.rerun()

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
                        answer = ask_groq(client, model_choice, context, question, build_patient_context_str())
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
        sex = st.session_state.pc_sex
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
                            m_question, build_patient_context_str()
                        )
                        st.markdown("### 🧾 Interpretation")
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"Groq API error: {e}")

st.markdown("---")
st.caption(
    "Built with Streamlit, PyMuPDF, Tesseract OCR, Sentence-Transformers, FAISS, and Groq. "
    "Reference ranges are general adult values compiled from widely-published clinical "
    "references and can vary by laboratory, analyzer, and method — always check the range "
    "printed on your actual report. Auto-detected patient details are a best-effort text "
    "match and should always be verified."
)
