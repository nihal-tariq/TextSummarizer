import nltk
import os

# Set up NLTK download path
NLTK_DATA_PATH = "/tmp/nltk_data"
nltk.data.path.append(NLTK_DATA_PATH)

# Download only if not already present
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=NLTK_DATA_PATH)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", download_dir=NLTK_DATA_PATH)

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", download_dir=NLTK_DATA_PATH)

try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4", download_dir=NLTK_DATA_PATH)


print(nltk.sent_tokenize("Hello there. This is a test."))



# --- MAIN IMPORTS ---
import streamlit as st
import PyPDF2
import docx
import re
from io import StringIO
from rake_nltk import Rake
import torch

try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
except ImportError as e:
    st.error(f"Missing dependencies! {str(e)}\n\nPlease install: pip install sentencepiece torch")
    st.stop()

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        return model, tokenizer
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}\n\nEnsure you have internet connection")
        st.stop()

model, tokenizer = load_model()

# --- FILE TEXT EXTRACTION ---
def extract_text(file):
    try:
        if file.name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(file)
            return "\n".join([page.extract_text() or "" for page in reader.pages])
        elif file.name.endswith(".txt"):
            return StringIO(file.getvalue().decode("utf-8")).read()
        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"File processing error: {str(e)}")
    return ""

# --- CHUNKING TEXT FOR SUMMARIZATION ---
def split_into_sections(text, max_chars=500):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) < max_chars:
            current += sentence + " "
        else:
            chunks.append(current.strip())
            current = sentence + " "
    if current:
        chunks.append(current.strip())
    return chunks

# --- FLASHCARD HEADING USING RAKE ---
def get_flashcard_heading(text):
    try:
        r = Rake(min_length=1, max_length=3)
        r.extract_keywords_from_text(text)
        keywords = r.get_ranked_phrases()
        if not keywords:
            sentences = nltk.sent_tokenize(text)
            return sentences[0][:70].strip() if sentences else "🔖 Untitled Section"
        best_keyword = keywords[0].replace('"', '').strip()
        return best_keyword[:70] or "🔖 Untitled Section"
    except Exception:
        return "🔖 Untitled Section"

# --- STREAMLIT UI ---
st.title("🧠 Flashcard Generator: Your new Study Partner")
option = st.radio("Choose input method:", ["📤 Upload a file", "✍️ Paste text"])

text = ""
if option == "📤 Upload a file":
    file = st.file_uploader("Upload a PDF, TXT, or DOCX", type=["pdf", "txt", "docx"])
    if file:
        text = extract_text(file)
        st.subheader("📜 Extracted Text Preview")
        st.caption(f"Character count: {len(text)}")
        st.write(text[:1000] + " [...]" if len(text) > 1000 else text)

elif option == "✍️ Paste text":
    text = st.text_area("Paste your long text here", height=200)

# Preprocessing
if text:
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.encode('ascii', 'ignore').decode()

# GENERATE FLASHCARDS
if st.button("📇 Generate Flashcards") and text.strip():
    st.subheader("🗂 Flashcards")
    sections = split_into_sections(text)

    if not sections:
        st.warning("No meaningful content found in the text.")
        st.stop()

    for i, chunk in enumerate(sections):
        with st.spinner(f"Processing section {i+1}/{len(sections)}..."):
            try:
                input_text = f"summarize: {chunk}"
                inputs = tokenizer(
                    input_text,
                    max_length=512,
                    truncation=True,
                    return_tensors="pt"
                )

                with torch.no_grad():
                    summary_ids = model.generate(
                        inputs.input_ids,
                        max_length=175,
                        min_length=75,
                        num_beams=4,
                        repetition_penalty=3.0,
                        length_penalty=2.0,
                        early_stopping=True,
                        temperature=0.9
                    )

                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                heading = get_flashcard_heading(summary)

                with st.expander(f"📌 {heading}"):
                    st.caption(f"Original text length: {len(chunk)} characters")
                    st.write(summary)
                    st.markdown("---")

                torch.cuda.empty_cache()

            except Exception as e:
                st.error(f"Section {i+1} error: {str(e)}")
                continue

st.markdown("---")
st.caption("Pro tip: Use well-structured text for best results! 🚀")
