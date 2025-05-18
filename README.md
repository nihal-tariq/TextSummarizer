# 🧠 Flashcard Generator & Summarizer

An AI-powered **flashcard generator** that helps you learn smarter by converting your PDFs, DOCX, or plain text into concise summaries and flashcards — all within an easy-to-use **Streamlit app**.

---

## ⚡️ Features

- **Multi-format input**: Upload `.pdf`, `.docx`, or `.txt` files, or simply paste your text.
- **Robust text extraction**: Uses `PyPDF2` and `python-docx` for accurate text retrieval.
- **Smart preprocessing & tokenization**: Cleans text, removes noise (URLs, HTML tags, extra spaces), and tokenizes using `nltk`.
- **Powerful summarization**: Integrates the `T5-small` model from Hugging Face Transformers for meaningful summaries.
- **Keyword extraction**: Uses `RAKE` to generate relevant flashcard headings from summaries.
- **Resource management**: Efficient downloading and caching of NLP resources and model weights to optimize performance.
- **User-friendly UI**: Built with Streamlit for interactive, easy-to-use web interface.

---

## 🚀 How It Works

1. **Upload** your file or paste your text.
2. The app **extracts and cleans** the text content.
3. Text is **split into manageable sections**.
4. Each section is **summarized** using the T5 model.
5. Keywords are extracted to create **flashcard headings**.
6. You get neat, expandable flashcards for quick revision.

---

## 🧰 Tech Stack

- Python 3.x  
- [Streamlit](https://streamlit.io/)  
- [NLTK](https://www.nltk.org/) for preprocessing and tokenization  
- [RAKE (Rapid Automatic Keyword Extraction)](https://pypi.org/project/rake-nltk/) for keyword extraction  
- [PyPDF2](https://pypi.org/project/PyPDF2/) for PDF text extraction  
- [python-docx](https://python-docx.readthedocs.io/en/latest/) for DOCX reading  
- [Transformers](https://huggingface.co/docs/transformers/) library (T5-small model) for summarization  

---

## 🧠 What I Learned

- Handling various document formats with different extraction libraries.  
- Implementing a robust text cleaning and tokenization pipeline using NLTK.  
- Integrating and fine-tuning Hugging Face models with Streamlit caching.  
- Managing resource downloads efficiently to optimize user experience.  
- Balancing summarization length and flashcard relevance through token limits and beam search tuning.

---




