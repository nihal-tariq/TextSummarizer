import streamlit as st
import nltk
import PyPDF2
import docx
import re
from io import StringIO
from rake_nltk import Rake

# NLTK Resource Handling - MUST BE AT THE TOP
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    st.warning("Downloading language resources (first-time setup)...")
    with st.spinner("This may take a minute..."):
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    st.rerun()

# Import handling for transformers
try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
except ImportError as e:
    st.error(f"Missing dependencies! {str(e)}\n\nPlease install: pip install sentencepiece torch")
    st.stop()


# Load model and tokenizer with enhanced caching
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


# Text extraction functions
def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    elif file.name.endswith(".txt"):
        return StringIO(file.getvalue().decode("utf-8")).read()
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""


def split_into_sections(text, max_chars=500):
    sections = re.split(r'(?:\n{2,}|\. )', text)  # Better splitting logic
    chunks = []
    current = ""
    for sec in sections:
        clean_sec = sec.strip()
        if len(clean_sec) < 30:
            continue
        if len(current) + len(clean_sec) < max_chars:
            current += clean_sec + " "
        else:
            chunks.append(current.strip())
            current = clean_sec + " "
    if current:
        chunks.append(current.strip())
    return chunks


# Improved keyword extraction
def get_flashcard_heading(text):
    try:
        r = Rake(min_length=1, max_length=3)  # Better keyword phrases
        r.extract_keywords_from_text(text)
        keywords = r.get_ranked_phrases()

        # Fallback to first meaningful sentence
        if not keywords:
            sentences = nltk.sent_tokenize(text)
            return sentences[0][:70].strip() if sentences else "Key Concept"

        best_keyword = keywords[0].replace('"', '').strip()
        return best_keyword[:70] or "Key Concept"
    except Exception as e:
        return "Important Topic"  # Silent fallback


# UI Components
st.title("🧠 Flashcard Generator from Documents")
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

# Enhanced text cleaning pipeline
if text:
    text = re.sub(r'http\S+', '', text)  # URLs
    text = re.sub(r'<[^>]+>', '', text)  # HTML tags
    text = re.sub(r'\n+', '\n', text)  # Extra newlines
    text = re.sub(r'\s+', ' ', text).strip()  # Whitespace
    text = text.encode('ascii', 'ignore').decode()  # Remove non-ASCII

if st.button("📇 Generate Flashcards") and text.strip():
    st.subheader("🗂 Flashcards")
    sections = split_into_sections(text)

    if not sections:
        st.warning("No meaningful content found in the text.")
        st.stop()

    for i, chunk in enumerate(sections):
        with st.spinner(f"Processing section {i + 1}/{len(sections)}..."):
            try:
                input_text = f"summarize: {chunk}"
                inputs = tokenizer(
                    input_text,
                    max_length=512,
                    truncation=True,
                    return_tensors="pt"
                )

                summary_ids = model.generate(
                    inputs.input_ids,
                    max_length=175,  # Increased for better context
                    min_length=75,
                    num_beams=4,
                    repetition_penalty=3.0,  # Stronger anti-repetition
                    length_penalty=2.0,
                    early_stopping=True,
                    temperature=0.9  # Slight randomness
                )

                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                heading = get_flashcard_heading(summary)

                with st.expander(f"📌 {heading}"):
                    st.caption(f"Original text length: {len(chunk)} characters")
                    st.write(summary)
                    st.markdown("---")

            except Exception as e:
                st.error(f"Section {i + 1} error: {str(e)}")
                continue

st.markdown("---")
st.caption("Pro tip: Use well-structured text for best results! 🚀")