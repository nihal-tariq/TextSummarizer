import nltk
import os
import streamlit as st
import PyPDF2
import docx
import re
from io import StringIO, BytesIO
from rake_nltk import Rake
import torch
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from transformers import T5ForConditionalGeneration, T5Tokenizer


NLTK_DATA_PATH = "/tmp/nltk_data"
nltk.data.path.append(NLTK_DATA_PATH)

for resource in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=NLTK_DATA_PATH)

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

def clean_text_for_keywords(text):
    text = re.sub(r'[()?/]', '', text)
    return text

def capitalize_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return ' '.join([s.strip().capitalize() for s in sentences])

def get_flashcard_heading(text):
    try:
        cleaned_text = clean_text_for_keywords(text)
        r = Rake(min_length=1, max_length=3)
        r.extract_keywords_from_text(cleaned_text)
        keywords = r.get_ranked_phrases()
        if not keywords:
            sentences = nltk.sent_tokenize(text)
            return sentences[0][:70].strip().title() if sentences else "🔖 Untitled Section"
        best_keyword = keywords[0].replace('"', '').strip().title()
        return best_keyword[:70].title() or "🔖 Untitled Section"
    except Exception:
        return "🔖 Untitled Section"

# UI Layout
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


if text:
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.encode('ascii', 'ignore').decode()

# Button to generate flashcards
if st.button("📇 Generate Flashcards") and text.strip():
    st.subheader("📚 Summary Preview")
    sections = split_into_sections(text)

    if not sections:
        st.warning("No meaningful content found in the text.")
        st.stop()

    flashcards = []
    combined_summary = ""

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
                        max_length=250,
                        min_length=100,
                        num_beams=5,
                        repetition_penalty=2.5,
                        length_penalty=1.5,
                        early_stopping=True,
                        temperature=0.8
                    )

                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summary = capitalize_sentences(summary)
                heading = get_flashcard_heading(summary)
                flashcards.append((heading, summary))
                combined_summary += f"{heading}:\n{summary}\n\n"

                with st.expander(f"📌 {heading}"):
                    st.caption(f"Original text length: {len(chunk)} characters")
                    st.write(summary)
                    st.markdown("---")

                torch.cuda.empty_cache()

            except Exception as e:
                st.error(f"Section {i+1} error: {str(e)}")
                continue

    if flashcards:

        st.subheader("📝 Full Combined Summary")
        st.text_area("Combined Summary", value=combined_summary.strip(), height=300)


        summary_pdf = BytesIO()
        c = canvas.Canvas(summary_pdf, pagesize=letter)
        width, height = letter
        y = height - 50
        c.setFont("Helvetica", 11)

        for line in combined_summary.strip().split('\n'):
            line = line.strip()
            if y < 60:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 11)
            c.drawString(50, y, line[:100]) 
            y -= 15

        c.save()
        summary_pdf.seek(0)

        st.download_button(
            label="📄 Download Summary Only as PDF",
            data=summary_pdf,
            file_name="summary.pdf",
            mime="application/pdf"
        )


        flashcard_pdf = BytesIO()
        c = canvas.Canvas(flashcard_pdf, pagesize=letter)
        width, height = letter
        y = height - 50

        for i, (heading, summary) in enumerate(flashcards, 1):
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, f"{i}. {heading}")
            y -= 20

            c.setFont("Helvetica", 10)
            for line in summary.split('. '):
                line = line.strip()
                if y < 60:
                    c.showPage()
                    y = height - 50
                    c.setFont("Helvetica", 10)
                c.drawString(60, y, f"- {line.strip()}.")
                y -= 15
            y -= 20

        c.save()
        flashcard_pdf.seek(0)

        st.download_button(
            label="📥 Download Flashcards as PDF",
            data=flashcard_pdf,
            file_name="flashcards.pdf",
            mime="application/pdf"
        )

st.markdown("---")
st.caption("Pro tip: Use well-structured text for best results! 🚀")
