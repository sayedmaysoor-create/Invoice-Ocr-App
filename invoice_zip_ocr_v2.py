import io
import os
import re
import zipfile
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from dateutil import parser as dateparser
import pytesseract

# Streamlit Cloud/Linux
pytesseract.pytesseract.tesseract_cmd = "tesseract"

st.set_page_config(page_title="Invoice ZIP OCR v2 (ENG+ARA)", layout="wide")
st.title("Invoice ZIP OCR v2 – Invoice Number + Due Date (English + Arabic)")

# ---------------- Helpers ----------------
def is_image_file(path: str) -> bool:
    ext = Path(path).suffix.lower()
    return ext in [".jpg", ".jpeg", ".png"]

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Upscale (helps small text)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Denoise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41, 11
    )
    return thresh

def normalize(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def ocr_text_and_confidence(cv_img: np.ndarray, lang: str = "eng+ara"):
    """
    Returns (text, avg_confidence, preview_text)
    avg_confidence is 0..100 (approx), based on Tesseract word confidences.
    """
    config = "--oem 3 --psm 6"

    # Full text
    text = pytesseract.image_to_string(cv_img, config=config, lang=lang)

    # Confidence (word-level)
    try:
        data = pytesseract.image_to_data(cv_img, config=config, lang=lang, output_type=pytesseract.Output.DICT)
        confs = []
        for c in data.get("conf", []):
            try:
                c = float(c)
                if c >= 0:
                    confs.append(c)
            except:
                pass
        avg_conf = float(np.mean(confs)) if confs else 0.0
    except:
        avg_conf = 0.0

    t = normalize(text)
    preview = (t[:600] + "…") if len(t) > 600 else t
    return t, avg_conf, preview

# ---------------- Field Extraction (ENG + ARA) ----------------
def find_invoice_number(text: str) -> str:
    # English patterns
    patterns = [
        r"(?:invoice\s*(?:no|number|#)\s*[:\-]?\s*)([A-Z0-9][A-Z0-9\-\/]+)",
        r"(?:inv\s*(?:no|#)\s*[:\-]?\s*)([A-Z0-9][A-Z0-9\-\/]+)",
        r"(?:invoice\s*[:\-]?\s*)([A-Z0-9][A-Z0-9\-\/]+)",
    ]
    # Arabic patterns (common labels)
    # رقم الفاتورة = invoice number
    # فاتورة رقم = invoice no
    patterns_ar = [
        r"(?:رقم\s*الفاتورة\s*[:\-]?\s*)([A-Z0-9][A-Z0-9\-\/]+)",
        r"(?:فاتورة\s*رقم\s*[:\-]?\s*)([A-Z0-9][A-Z0-9\-\/]+)",
        r"(?:رقم\s*فاتورة\s*[:\-]?\s*)([A-Z0-9][A-Z0-9\-\/]+)",
    ]

    for p in patterns + patterns_ar:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""

def parse_date_safe(raw: str) -> str:
    if not raw:
        return ""
    try:
        # fuzzy=True lets it handle "Due Date: 12/03/2025" etc.
        dt = dateparser.parse(raw, fuzzy=True)
        return dt.date().isoformat()
    except:
        return ""

def find_due_date(text: str) -> str:
    # English patterns
    patterns = [
        r"(?:due\s*date\s*[:\-]?\s*)([A-Za-z0-9,\/\.\-\s]+)",
        r"(?:payment\s*due\s*[:\-]?\s*)([A-Za-z0-9,\/\.\-\s]+)",
        r"(?:due\s*on\s*[:\-]?\s*)([A-Za-z0-9,\/\.\-\s]+)",
    ]
    # Arabic patterns
    # تاريخ الاستحقاق = due date
    # تاريخ الاستحقاق/الدفع/الاستلام varies by vendor
    patterns_ar = [
        r"(?:تاريخ\s*الاستحقاق\s*[:\-]?\s*)([A-Za-z0-9,\/\.\-\s]+)",
        r"(?:تاريخ\s*استحقاق\s*[:\-]?\s*)([A-Za-z0-9,\/\.\-\s]+)",
        r"(?:تاريخ\s*الدفع\s*[:\-]?\s*)([A-Za-z0-9,\/\.\-\s]+)",
    ]

    for p in patterns + patterns_ar:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            parsed = parse_date_safe(raw)
            return parsed if parsed else raw  # fallback to raw if parsing fails
    return ""

def score_extraction(invoice_no: str, due_date: str, ocr_conf: float) -> float:
    """
    Simple confidence score 0..100 based on:
    - OCR confidence
    - whether invoice number is present
    - whether due date is present and parsed well (YYYY-MM-DD)
    """
    score = 0.0
    # OCR contributes up to 60 points
    score += min(max(ocr_conf, 0.0), 100.0) * 0.60

    # Fields contribute up to 40 points
    if invoice_no:
        score += 20.0
    if due_date:
        score += 10.0
        if re.match(r"^\d{4}-\d{2}-\d{2}$", due_date):
            score += 10.0  # parsed clean ISO date
    return round(min(score, 100.0), 1)

# ---------------- UI ----------------
st.sidebar.header("Batch Settings")
batch_size = st.sidebar.number_input("Batch size (per click)", min_value=10, max_value=500, value=100, step=10)
review_threshold = st.sidebar.slider("Mark Needs Review if score below", min_value=10, max_value=90, value=55, step=5)
lang_mode = st.sidebar.selectbox("OCR Language", ["eng+ara (English + Arabic)", "eng (English only)", "ara (Arabic only)"])

lang_map = {
    "eng+ara (English + Arabic)": "eng+ara",
    "eng (English only)": "eng",
    "ara (Arabic only)": "ara"
}

zip_file = st.file_uploader("Upload a ZIP containing invoice images (JPG/PNG)", type=["zip"])

# Keep state between button clicks (batch processing)
if "all_images" not in st.session_state:
    st.session_state.all_images = []
if "results" not in st.session_state:
    st.session_state.results = []
if "cursor" not in st.session_state:
    st.session_state.cursor = 0
if "zip_name" not in st.session_state:
    st.session_state.zip_name = ""

def reset_state():
    st.session_state.all_images = []
    st.session_state.results = []
    st.session_state.cursor = 0
    st.session_state.zip_name = ""

if zip_file is None:
    st.info("Upload a ZIP to start. For 2,000 invoices, batch mode is the safest.")
else:
    # If user uploads a different ZIP, reset
    if st.session_state.zip_name != zip_file.name:
        reset_state()
        st.session_state.zip_name = zip_file.name

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "invoices.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_file.getbuffer())

            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(tmpdir)

            all_files = []
            for root, _, files in os.walk(tmpdir):
                for name in files:
                    full_path = os.path.join(root, name)
                    if is_image_file(full_path):
                        # Copy paths into memory by reading bytes now (since tmpdir will be deleted)
                        with open(full_path, "rb") as imgf:
                            all_files.append((os.path.relpath(full_path, tmpdir), imgf.read()))

            all_files.sort(key=lambda x: x[0])
            st.session_state.all_images = all_files

    total_images = len(st.session_state.all_images)
    st.write(f"ZIP: **{st.session_state.zip_name}**")
    st.write(f"Images found: **{total_images}**")
    st.write(f"Processed so far: **{len(st.session_state.results)}** | Next index: **{st.session_state.cursor}**")

    colA, colB, colC = st.columns([1, 1, 2])

    with colA:
        process_next = st.button("Process next batch ✅", use_container_width=True)
    with colB:
        reset_btn = st.button("Reset / Start over 🔄", use_container_width=True)
    with colC:
        st.caption("Tip: If your invoices are mixed layouts, you’ll get best results by reviewing the flagged ones only.")

    if reset_btn:
        reset_state()
        st.rerun()

    if process_next:
        start = st.session_state.cursor
        end = min(start + int(batch_size), total_images)

        if start >= total_images:
            st.warning("All images are already processed.")
        else:
            progress = st.progress(0)
            status = st.empty()

            lang = lang_map[lang_mode]

            for idx, (rel_name, img_bytes) in enumerate(st.session_state.all_images[start:end], start=1):
                try:
                    pil_img = Image.open(io.BytesIO(img_bytes))
                    cv_img = preprocess_image(pil_img)

                    text, ocr_conf, preview = ocr_text_and_confidence(cv_img, lang=lang)

                    invoice_no = find_invoice_number(text)
                    due_date = find_due_date(text)

                    score = score_extraction(invoice_no, due_date, ocr_conf)
                    needs_review = (score < float(review_threshold)) or (invoice_no == "") or (due_date == "")

                    st.session_state.results.append({
                        "FileName": rel_name,
                        "InvoiceNumber": invoice_no,
                        "DueDate": due_date,
                        "OCR_Conf": round(float(ocr_conf), 1),
                        "Score": score,
                        "NeedsReview": needs_review,
                        "OCR_Text_Preview": preview if needs_review else ""  # show text only when flagged
                    })

                except Exception as e:
                    st.session_state.results.append({
                        "FileName": rel_name,
                        "InvoiceNumber": "",
                        "DueDate": "",
                        "OCR_Conf": 0.0,
                        "Score": 0.0,
                        "NeedsReview": True,
                        "OCR_Text_Preview": "OCR failed for this image."
                    })

                progress.progress(idx / (end - start))
                status.write(f"Processing {start + idx} / {total_images}")

            st.session_state.cursor = end
            st.success(f"Batch done: processed {start} → {end-1}")

    # Show results
    if st.session_state.results:
        df = pd.DataFrame(st.session_state.results)

        st.divider()
        st.subheader("Results (all processed)")
        st.dataframe(df[["FileName", "InvoiceNumber", "DueDate", "OCR_Conf", "Score", "NeedsReview"]], use_container_width=True)

        flagged = df[df["NeedsReview"] == True].copy()
        st.write(f"⚠️ **Needs Review:** {len(flagged)} out of {len(df)}")

        st.subheader("Needs Review (shows OCR text preview)")
        if len(flagged) == 0:
            st.success("Nothing flagged. Nice.")
        else:
            st.dataframe(flagged[["FileName", "InvoiceNumber", "DueDate", "OCR_Conf", "Score", "OCR_Text_Preview"]],
                         use_container_width=True)

        # Download CSV (without OCR preview column, unless you want it)
        export_df = df[["FileName", "InvoiceNumber", "DueDate", "OCR_Conf", "Score", "NeedsReview"]]
        st.download_button(
            "Download CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="invoice_numbers_due_dates_v2.csv",
            mime="text/csv"
        )
    else:
        st.info("Click **Process next batch ✅** to start.")
