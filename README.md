# 🩺 AssistDoc - AI Medical Assistant

AssistDoc is an AI-powered medical assistant designed to help users understand their symptoms, receive preliminary disease suggestions, retrieve reliable medical information, and locate nearby hospitals. The system combines Natural Language Processing (NLP), Information Retrieval, and Large Language Models (LLMs) to provide an interactive healthcare support experience.

Disclaimer: AssistDoc is intended for educational and informational purposes only. It is not a substitute for professional medical diagnosis or treatment.



# 🚀 Features

- 🤖 AI-powered medical chatbot
- 🩺 Multi-turn symptom assessment
- 🔍 Disease prediction based on symptoms
- 📚 PDF-based medical knowledge retrieval (Hybrid RAG)
- 🧠 TF-IDF and Cosine Similarity for symptom matching
- 💬 LLM-powered medical responses using Groq
- 🏥 Nearby hospital recommendations using Google Maps API
- 📅 Appointment booking interface
- 🎤 Speech-to-Text and Text-to-Speech support
- 📄 Medical PDF knowledge base
- 📱 Responsive and modern UI

---

# 🏗️ System Architecture

```
                User
                  │
                  ▼
         AssistDoc Frontend
                  │
                  ▼
          FastAPI Backend
                  │
      ┌───────────┴───────────┐
      │                       │
      ▼                       ▼
Symptom Processing      Medical PDF
      │                  Knowledge Base
      ▼                       │
TF-IDF Vectorization          │
      │                       │
Cosine Similarity      Keyword Retrieval
      │                       │
      └───────────┬───────────┘
                  ▼
            Groq LLM API
                  │
                  ▼
          AI Medical Response
                  │
                  ▼
 Hospital Recommendation System
```

---

# 🛠️ Technologies Used

## Backend

- Python
- FastAPI
- NumPy
- Pandas
- Scikit-Learn
- PyPDF2
- Groq API
- HTTPX

## Frontend

- HTML5
- CSS3
- JavaScript
- Typed.js
- AOS Animation Library

## AI & NLP

- TF-IDF Vectorizer
- Cosine Similarity
- Label Encoder
- Hybrid Retrieval-Augmented Generation (RAG)

## APIs

- Google Maps API
- Groq LLM API

---

# 📂 Project Structure

```
AssistDoc/
│
├── app.py
├── symptoms_dataset.csv
├── appointment.html
├── templates/
│     ├── main_index.html
│     └── drpeppy.mp4
│
├── pdfs/
│     └── your_pdf.pdf
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/AssistDoc.git

cd AssistDoc
```

---

## Create Virtual Environment

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux / Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment Variables

Create a `.env` file.

```env
GROQ_API_KEY=YOUR_GROQ_KEY

GOOGLE_MAPS_API_KEY=YOUR_GOOGLE_MAPS_KEY

HF_API_KEY=YOUR_HUGGINGFACE_KEY
```

---

## Run the Application

```bash
uvicorn app:app --reload
```

Application URL

```
http://127.0.0.1:8000
```

---

# 🧠 Working Flow

1. User enters symptoms.
2. Symptoms are normalized.
3. TF-IDF converts symptoms into vectors.
4. Cosine Similarity retrieves similar diseases.
5. Follow-up questions improve prediction.
6. Relevant medical PDF sections are retrieved.
7. Groq LLM generates the response.
8. Nearby hospitals are suggested.
9. User can book appointments.

---

# 📚 Hybrid RAG Workflow

```
User Query
      │
      ▼
Medical PDF
      │
Chunking
      │
Keyword Retrieval
      │
Relevant Chunks
      │
Groq LLM
      │
Generated Medical Response
```

---

# 🏥 Hospital Recommendation

AssistDoc integrates Google Maps API to:

- Find nearby hospitals
- Display ratings
- Show distance
- Display consultation fees
- View hospital details
- Book appointments

---

# 📊 AI Techniques Used

- TF-IDF Vectorization
- Cosine Similarity
- Natural Language Processing
- Symptom Normalization
- Multi-turn Question Answering
- Retrieval-Augmented Generation (Hybrid)
- Large Language Models

---

# 📈 Future Enhancements

- Vector Database (FAISS/Pinecone)
- Sentence Transformer Embeddings
- Electronic Health Record Integration
- Voice Assistant
- Image-based Disease Detection
- Medical Report Summarization
- Wearable Device Integration
- Multi-language Support
- Doctor Dashboard

---




# 📄 License

This project is developed for educational and academic purposes.

---

# ⭐ Acknowledgements

- FastAPI
- Groq
- Google Maps Platform
- Scikit-Learn
- PyPDF2
- Python Community
