import os
import uuid
import json
import random
import urllib.parse
from datetime import datetime
from typing import List, Optional, Dict, Any
import asyncio
import threading

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import groq
from dotenv import load_dotenv
import PyPDF2

# Load environment variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_INDEX_PATH = os.path.join(BASE_DIR, "templates", "main_index.html")
APPOINTMENT_PATH = os.path.join(BASE_DIR, "appointment.html")
DRPEPPY_PATH = os.path.join(BASE_DIR, "templates", "drpeppy.mp4")

HF_API_KEY = os.getenv("HF_API_KEY", "")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_pdfs")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="AssistDoc Backend", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory stores (replace with DB in production)
SESSIONS: Dict[str, Dict[str, Any]] = {}
APPOINTMENTS: List[Dict[str, Any]] = []

DATASET_PATH = os.path.join(BASE_DIR, "symptoms_dataset.csv")
RAG_META_PATH = os.path.join(BASE_DIR, "rag_meta.json")
RAG_EMB_PATH = os.path.join(BASE_DIR, "rag_embeddings.npy")
PDF_PATH = os.path.join(BASE_DIR, "pdfs", "your_pdf.pdf")

# Initialize Groq client for LLM-based predictions
groq_client = None
if GROQ_API_KEY:
    groq_client = groq.Client(api_key=GROQ_API_KEY)

PDF_CONTENT = ""
PDF_CHUNKS = []
INDEXING_IN_PROGRESS = False

def load_pdf_content():
    """Load and extract text from the PDF file"""
    global PDF_CONTENT, PDF_CHUNKS
    try:
        if os.path.exists(PDF_PATH):
            with open(PDF_PATH, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text_content = []
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text_content.append(page.extract_text())
                PDF_CONTENT = "\n".join(text_content)
                print(f"[AssistDoc] Successfully loaded PDF from {PDF_PATH}")
                print(f"[AssistDoc] PDF content length: {len(PDF_CONTENT)} characters")
                
                create_pdf_chunks()
        else:
            print(f"[AssistDoc] PDF file not found at {PDF_PATH}")
            PDF_CONTENT = ""
    except Exception as e:
        print(f"[AssistDoc] Error loading PDF: {e}")
        PDF_CONTENT = ""

def create_pdf_chunks():
    """Split PDF content into semantic chunks"""
    global PDF_CHUNKS
    try:
        # Split by paragraphs first
        paragraphs = PDF_CONTENT.split('\n\n')
        PDF_CHUNKS = []
        
        for para in paragraphs:
            para = para.strip()
            if not para or len(para) < 20:
                continue
            
            # If paragraph is very long, split by sentences
            if len(para) > 500:
                sentences = []
                current = ""
                for char in para:
                    current += char
                    if char in '.!?':
                        sentences.append(current.strip())
                        current = ""
                if current.strip():
                    sentences.append(current.strip())
                
                # Group sentences into chunks of 2-3 sentences
                for i in range(0, len(sentences), 2):
                    chunk = " ".join(sentences[i:i+3]).strip()
                    if len(chunk) > 20:
                        PDF_CHUNKS.append(chunk)
            else:
                PDF_CHUNKS.append(para)
        
        print(f"[AssistDoc] Created {len(PDF_CHUNKS)} PDF chunks")
    except Exception as e:
        print(f"[AssistDoc] Error creating PDF chunks: {e}")
        PDF_CHUNKS = []

# Load dataset and prepare ML model
try:
    df_dataset = pd.read_csv(DATASET_PATH)
    if not df_dataset.empty and 'symptoms' in df_dataset.columns:
        df_dataset['symptoms'] = df_dataset['symptoms'].fillna('').astype(str)
        # Create TF-IDF vectorizer for symptom matching with n-grams for better phrase matching
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
        symptoms_tfidf = vectorizer.fit_transform(df_dataset['symptoms'])
        # Create label encoder for disease names
        label_encoder = LabelEncoder()
        if 'disease' in df_dataset.columns:
            df_dataset['disease'] = df_dataset['disease'].fillna('Unknown').astype(str)
            label_encoder.fit(df_dataset['disease'])
    else:
        vectorizer = None
        symptoms_tfidf = None
        label_encoder = None
except Exception as e:
    print(f"Error initializing ML models: {e}")
    df_dataset = pd.DataFrame()
    symptoms_tfidf = None
    vectorizer = None
    label_encoder = None

load_pdf_content()

# Symptom processing config
MIN_SYMPTOM_THRESHOLD = 3

SYMPTOM_SYNONYMS = {
    "broken bone": "fracture",
    "bone broken": "fracture",
    "bone crack": "fracture",
    "cracked bone": "fracture",
    "sprain": "ligament injury",
    "shortness of breath": "dyspnea",
    "tummy pain": "abdominal pain",
    "stomach ache": "abdominal pain",
    "sore throat": "throat pain",
    "runny nose": "nasal discharge",
    "joint ache": "joint pain",
    "muscle ache": "muscle pain",
    "body ache": "body pain",
    "flu": "influenza",
    "cold": "upper respiratory infection",
    "migraine": "headache",
    "migraine headache": "severe headache",
    "heart attack": "acute myocardial infarction",
    "heart disease": "cardiac condition",
    "high blood pressure": "hypertension",
    "low blood pressure": "hypotension",
    "high fever": "fever",
    "low fever": "fever",
    "severe pain": "pain",
    "mild pain": "pain",
    "moderate pain": "pain",
    "extreme pain": "severe pain",
    "difficulty breathing": "dyspnea",
    "trouble breathing": "dyspnea",
    "hard to breathe": "dyspnea",
    "can't breathe": "dyspnea",
    "throwing up": "vomiting",
    "feeling sick": "nausea",
    "stomach upset": "gastric distress",
    "belly pain": "abdominal pain",
    "tummy ache": "abdominal pain",
    "skin rash": "rash",
    "itchy skin": "pruritus",
    "red skin": "erythema",
    "swollen joints": "joint swelling",
    "stiff joints": "joint stiffness",
    "weak muscles": "muscle weakness",
    "muscle weakness": "myasthenia",
    "tired": "fatigue",
    "exhausted": "fatigue",
    "dizzy": "dizziness",
    "lightheaded": "dizziness",
    "faint": "syncope",
    "fainting": "syncope",
    "blurred vision": "vision problems",
    "vision changes": "vision problems",
    "eye pain": "ocular pain",
    "ear pain": "otalgia",
    "ear ache": "otalgia",
    "hearing loss": "deafness",
    "ringing in ears": "tinnitus",
    "nose bleed": "epistaxis",
    "nosebleed": "epistaxis",
    "blood in stool": "hematochezia",
    "blood in urine": "hematuria",
    "painful urination": "dysuria",
    "frequent urination": "polyuria",
    "urinary urgency": "urinary frequency",
    "constipation": "bowel obstruction",
    "diarrhea": "loose stool",
    "loose stool": "diarrhea",
    "watery stool": "diarrhea",
    "bloody stool": "hematochezia",
    "black stool": "melena",
    "pale skin": "pallor",
    "yellow skin": "jaundice",
    "yellowing": "jaundice",
    "swelling": "edema",
    "puffiness": "edema",
    "inflammation": "inflammatory response",
    "inflamed": "inflammation",
    "bruising": "contusion",
    "bruise": "contusion",
    "wound": "laceration",
    "cut": "laceration",
    "burn": "thermal injury",
    "sunburn": "solar burn",
    "chills": "rigors",
    "sweating": "diaphoresis",
    "night sweats": "nocturnal diaphoresis",
    "weight loss": "weight decrease",
    "weight gain": "weight increase",
    "loss of appetite": "anorexia",
    "no appetite": "anorexia",
    "increased appetite": "polyphagia",
    "thirst": "polydipsia",
    "excessive thirst": "polydipsia",
    "chest pain": "chest pain",
    "heart pain": "chest pain",
    "stomach pain": "abdominal pain",
    "belly pain": "abdominal pain",
    "back pain": "back pain",
    "neck pain": "neck pain",
    "head pain": "headache",
    "ear pain": "ear pain",
    "tooth pain": "dental pain",
    "toothache": "dental pain",
    "joint pain": "arthralgia",
    "muscle pain": "myalgia",
    "skin pain": "cutaneous pain",
    "breathing difficulty": "dyspnea",
    "short of breath": "dyspnea",
    "cough": "cough",
    "fever": "fever",
    "chills": "chills",
    "nausea": "nausea",
    "vomiting": "vomiting",
    "diarrhea": "diarrhea",
    "constipation": "constipation",
    "rash": "rash",
    "itching": "pruritus",
    "swelling": "edema",
    "dizziness": "dizziness",
    "fatigue": "fatigue",
    "weakness": "asthenia",
    "dry mouth": "xerostomia",
    "mouth sores": "oral ulcers",
    "tongue pain": "glossalgia",
    "gum pain": "gingival pain",
    "tooth pain": "dental pain",
    "toothache": "dental pain",
    "jaw pain": "temporomandibular pain",
    "neck pain": "cervical pain",
    "neck stiffness": "nuchal rigidity",
    "back pain": "dorsal pain",
    "lower back pain": "lumbar pain",
    "upper back pain": "thoracic pain",
    "shoulder pain": "shoulder ache",
    "arm pain": "arm ache",
    "leg pain": "leg ache",
    "knee pain": "knee ache",
    "ankle pain": "ankle ache",
    "foot pain": "foot ache",
    "hand pain": "hand ache",
    "wrist pain": "wrist ache",
    "elbow pain": "elbow ache",
    "hip pain": "hip ache",
    "groin pain": "inguinal pain",
    "pelvic pain": "pelvic ache",
    "menstrual pain": "dysmenorrhea",
    "period pain": "dysmenorrhea",
    "cramps": "muscle cramps",
    "muscle cramps": "myalgia",
    "spasm": "muscle spasm",
    "tremor": "trembling",
    "shaking": "tremor",
    "numbness": "paresthesia",
    "tingling": "paresthesia",
    "pins and needles": "paresthesia",
    "weakness": "asthenia",
    "paralysis": "motor paralysis",
    "loss of consciousness": "syncope",
    "confusion": "delirium",
    "memory loss": "amnesia",
    "forgetfulness": "memory impairment",
    "difficulty concentrating": "cognitive impairment",
    "anxiety": "anxiety disorder",
    "panic": "panic attack",
    "depression": "depressive disorder",
    "sadness": "depression",
    "mood swings": "mood instability",
    "irritability": "irritable mood",
    "anger": "anger management issue",
    "insomnia": "sleep disorder",
    "sleeplessness": "insomnia",
    "can't sleep": "insomnia",
    "trouble sleeping": "sleep disturbance",
    "excessive sleeping": "hypersomnia",
    "sleepiness": "somnolence",
    "snoring": "sleep apnea",
    "sleep apnea": "obstructive sleep apnea",
    "nightmares": "sleep disturbance",
    "night terrors": "sleep disturbance",
    "sleepwalking": "somnambulism",
    "hallucinations": "hallucinosis",
    "delusions": "delusional disorder",
    "paranoia": "paranoid ideation",
    "obsessive thoughts": "obsessive-compulsive disorder",
    "compulsive behavior": "obsessive-compulsive disorder",
    "phobia": "anxiety disorder",
    "fear": "anxiety",
    "stress": "psychological stress",
    "tension": "muscle tension",
    "nervousness": "anxiety",
    "restlessness": "agitation",
    "hyperactivity": "attention deficit hyperactivity disorder",
    "impulsivity": "impulse control disorder",
    "aggression": "aggressive behavior",
    "violence": "violent behavior",
    "self-harm": "self-injurious behavior",
    "suicidal thoughts": "suicidality",
    "substance abuse": "addiction",
    "alcohol abuse": "alcoholism",
    "drug abuse": "drug addiction",
    "smoking": "tobacco use",
    "nicotine addiction": "tobacco dependence",
    "caffeine addiction": "caffeine dependence",
    "food addiction": "eating disorder",
    "gambling addiction": "gambling disorder",
    "internet addiction": "internet gaming disorder",
    "sex addiction": "sexual addiction",
    "work addiction": "workaholism",
    "exercise addiction": "compulsive exercise",
    "shopping addiction": "compulsive shopping",
}

def normalize_symptoms_text(text: str) -> str:
    t = (text or "").lower()
    for k, v in SYMPTOM_SYNONYMS.items():
        t = t.replace(k, v)
    return t

def estimate_symptom_count(texts: List[str]) -> int:
    """Heuristic: count symptom phrases separated by commas/and across accumulated messages."""
    combined = " ".join(texts).lower()
    # Split on commas and ' and '
    parts = []
    for chunk in combined.split(","):
        parts.extend([p.strip() for p in chunk.split(" and ")] if chunk.strip() else [])
    # Filter out very short fragments
    parts = [p for p in parts if len(p.split()) >= 2]
    return max(1, len(parts)) if parts else 0

def search_pdf_fallback(query: str, top_k: int = 3) -> List[str]:
    """Keyword-based PDF search"""
    if not PDF_CHUNKS:
        print("[AssistDoc] PDF_CHUNKS is empty!")
        return []
    
    try:
        print(f"[AssistDoc] Using keyword search for: '{query}'")
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'what', 'which', 'who', 'when', 'where', 'why', 'how', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'if', 'it', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        query_words = query_words - stop_words
        
        scored_chunks = []
        for chunk_idx, chunk in enumerate(PDF_CHUNKS):
            chunk_lower = chunk.lower()
            keyword_matches = sum(1 for word in query_words if word in chunk_lower)
            
            if keyword_matches > 0:
                score = keyword_matches * 3
                scored_chunks.append((score, chunk_idx, chunk))
        
        if not scored_chunks:
            print(f"[AssistDoc] No matching chunks found, returning first {top_k}")
            return PDF_CHUNKS[:top_k]
        
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        results = [chunk for score, idx, chunk in scored_chunks[:top_k]]
        
        print(f"[AssistDoc] Keyword search returned {len(results)} results")
        return results
        
    except Exception as e:
        print(f"[AssistDoc] Error in keyword search: {e}")
        return []

# ----------------------
# Multi-turn question bank
# ----------------------
QUESTION_BANK: Dict[str, List[Dict[str, Any]]] = {
    "fever": [
        {"key": "fever_level", "text": "How high is your fever?", "options": ["Low (99-100°F)", "Medium (100-102°F)", "High (102°F+)"]},
        {"key": "fever_duration", "text": "How long have you had the fever?", "options": ["< 1 day", "1-3 days", "> 3 days"]},
        {"key": "fever_associated", "text": "Any associated symptoms?", "options": ["Just fever", "Headache & Body Aches", "Cough & Sore Throat", "Stomach pain & Nausea"]},
    ],
    "headache": [
        {"key": "headache_location", "text": "Where is the pain located?", "options": ["Forehead/Temples", "Back of Head", "All over"]},
        {"key": "pain_intensity", "text": "Rate pain intensity (1-10)", "options": ["1-3 (Mild)", "4-6 (Moderate)", "7-10 (Severe)"]},
        {"key": "sudden_onset", "text": "Sudden severe or gradual?", "options": ["Sudden and severe", "Gradual onset"]},
    ],
    "cough": [
        {"key": "cough_type", "text": "Dry or with mucus?", "options": ["Dry", "Mucus"]},
        {"key": "cough_duration", "text": "How long?", "options": ["< 1 week", "1-3 weeks", "> 1 month"]},
        {"key": "respiratory_difficulty", "text": "Shortness of breath or chest pain?", "options": ["No", "Shortness of breath", "Chest pain"]},
    ],
    "stomach pain": [
        {"key": "pain_location", "text": "Where is the pain?", "options": ["Upper Abdomen", "Lower Right Abdomen", "Lower Left Abdomen", "All over"]},
        {"key": "pain_type", "text": "Describe the pain", "options": ["Sharp", "Dull", "Cramping", "Burning"]},
        {"key": "associated_symptoms", "text": "Nausea/vomiting or bowel changes?", "options": ["No", "Nausea/Vomiting", "Diarrhea/Constipation"]},
    ],
    "chest pain": [
        {"key": "pain_descriptor", "text": "Pain character?", "options": ["Sharp (Stabbing)", "Pressure (Squeezing)", "Burning"]},
        {"key": "pain_radiation", "text": "Does it radiate?", "options": ["No", "Arm/Shoulder", "Neck/Jaw"]},
        {"key": "associated_symptoms_chest", "text": "Shortness of breath or sweating?", "options": ["No", "Shortness of breath", "Sweating + Shortness of breath"]},
    ],
    "rash": [
        {"key": "rash_location", "text": "Where is the rash located?", "options": ["Face/Neck", "Trunk/Torso", "Arms/Legs", "Whole body"]},
        {"key": "rash_appearance", "text": "How does the rash look?", "options": ["Red and flat", "Raised bumps", "Blisters", "Itchy patches"]},
        {"key": "rash_duration", "text": "How long have you had the rash?", "options": ["< 1 day", "1-3 days", "> 3 days"]},
    ],
    "joint pain": [
        {"key": "joint_location", "text": "TEST: Which specific joints are affected? (e.g., knees, fingers, wrists)", "options": ["Knees", "Fingers/Hands", "Wrists", "Ankles", "Elbows", "Shoulders", "Multiple joints", "Other"]},
        {"key": "joint_swelling", "text": "Is there any swelling, redness, or warmth in the affected joints?", "options": ["No swelling", "Mild swelling", "Significant swelling with redness", "Severe swelling and warmth"]},
        {"key": "joint_pain_type", "text": "What type of pain is it?", "options": ["Sharp/stabbing", "Dull/aching", "Throbbing", "Burning"]},
        {"key": "joint_duration", "text": "How long have you had this joint pain?", "options": ["Less than 1 week", "1-4 weeks", "1-6 months", "More than 6 months"]},
        {"key": "joint_morning_stiffness", "text": "Do you experience morning stiffness in the joints?", "options": ["No", "Less than 30 minutes", "30-60 minutes", "More than 1 hour"]},
    ],
    "fatigue": [
        {"key": "fatigue_onset", "text": "How did the fatigue start?", "options": ["Suddenly", "Gradually over days", "Gradually over weeks"]},
        {"key": "fatigue_severity", "text": "How severe is your fatigue?", "options": ["Mild - can do most activities", "Moderate - limiting some activities", "Severe - can barely function"]},
        {"key": "fatigue_relief", "text": "Does rest improve your fatigue?", "options": ["Yes, completely", "Somewhat", "Not at all"]},
    ],
    "abdominal pain": [
        {"key": "abdominal_location", "text": "Which part of your abdomen hurts?", "options": ["Upper", "Lower", "Right side", "Left side", "Center"]},
        {"key": "abdominal_timing", "text": "Is the pain constant or intermittent?", "options": ["Constant", "Comes and goes"]},
        {"key": "abdominal_severity", "text": "How would you rate the pain (1-10)?", "options": ["1-3 (Mild)", "4-6 (Moderate)", "7-10 (Severe)"]},
        {"key": "abdominal_meal_relation", "text": "Is the pain related to meals?", "options": ["Worse after eating", "Worse when hungry", "No relation to food"]},
        {"key": "abdominal_other_symptoms", "text": "Do you have any other symptoms?", "options": ["Nausea", "Vomiting", "Fever", "Diarrhea", "None"]},
    ],
    "stomach ache": [
        {"key": "stomach_location", "text": "Where is the stomach pain located?", "options": ["Upper abdomen", "Lower abdomen", "All over abdomen"]},
        {"key": "stomach_timing", "text": "When does the pain occur?", "options": ["After meals", "On empty stomach", "No specific pattern"]},
        {"key": "stomach_duration", "text": "How long have you had this pain?", "options": ["Less than a day", "Few days", "A week or more"]},
        {"key": "stomach_severity", "text": "How severe is the pain?", "options": ["Mild", "Moderate", "Severe"]},
        {"key": "stomach_associated", "text": "Any associated symptoms?", "options": ["Nausea", "Vomiting", "Bloating", "Fever", "None"]},
    ],
    "edema": [
        {"key": "edema_location", "text": "Where is the swelling located?", "options": ["Face/Eyes", "Arms", "Legs", "Abdomen", "Whole body", "Other"]},
        {"key": "edema_onset", "text": "When did the swelling start?", "options": ["Suddenly", "Gradually over hours", "Gradually over days"]},
        {"key": "edema_pain", "text": "Is the swelling painful?", "options": ["No pain", "Mild discomfort", "Moderate pain", "Severe pain"]},
        {"key": "edema_changes", "text": "Does it change with position or time of day?", "options": ["Worse in morning", "Worse in evening", "Worse when standing", "No change"]},
        {"key": "edema_associated", "text": "Any associated symptoms?", "options": ["Shortness of breath", "Weight gain", "Fatigue", "Skin changes", "None"]},
    ],
    "default": [
        {"key": "symptom_duration", "text": "How long have you had this symptom?", "options": ["Less than 24 hours", "1-3 days", "4-7 days", "More than a week"]},
        {"key": "symptom_severity", "text": "How severe is this symptom?", "options": ["Mild", "Moderate", "Severe", "Extreme"]},
        {"key": "symptom_progression", "text": "Is this symptom getting better, worse, or staying the same?", "options": ["Getting better", "Getting worse", "Staying the same", "Fluctuating"]},
    ]
}

FOLLOW_UPS: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    "fever_associated": {
        "Headache & Body Aches": [{"key": "stiff_neck", "text": "Stiff neck or light sensitivity?", "options": ["No", "Stiff neck", "Light sensitivity"]}],
        "Stomach pain & Nausea": [{"key": "rlq_focus", "text": "Pain mainly in lower right abdomen?", "options": ["No", "Yes"]}],
        "Cough & Sore Throat": [{"key": "breathing_difficulty", "text": "Any difficulty breathing?", "options": ["No", "Mild", "Significant"]}],
    },
    "pain_intensity": {
        "7-10 (Severe)": [{"key": "visual_changes", "text": "Any visual changes or vomiting?", "options": ["No", "Visual changes", "Vomiting"]}],
    },
    "abdominal_location": {
        "Upper": [{"key": "upper_abdominal_type", "text": "What type of pain is it?", "options": ["Burning", "Sharp", "Dull/aching"]}],
        "Lower": [{"key": "lower_abdominal_duration", "text": "How long have you had this pain?", "options": ["Hours", "Days", "Weeks"]}],
        "Right side": [{"key": "right_side_worse", "text": "Is it worse with movement?", "options": ["Yes", "No", "Not sure"]}],
    },
    "stomach_location": {
        "Upper abdomen": [{"key": "upper_stomach_character", "text": "How would you describe the pain?", "options": ["Burning", "Gnawing", "Sharp"]}],
        "Lower abdomen": [{"key": "lower_stomach_bowel", "text": "Any changes in bowel movements?", "options": ["Normal", "Diarrhea", "Constipation"]}],
    },
    "stomach_ache_location": {
        "Upper abdomen": [{"key": "upper_ache_meal", "text": "Is it worse after specific foods?", "options": ["No", "Spicy foods", "Fatty foods", "All foods"]}],
        "Lower abdomen": [{"key": "lower_ache_relief", "text": "What provides relief?", "options": ["Nothing helps", "Antacids", "Bowel movement", "Position change"]}],
    },
    "respiratory_difficulty": {
        "Shortness of breath": [{"key": "breathing_duration", "text": "How long have you had breathing difficulty?", "options": ["Recent (hours)", "Days", "Weeks or longer"]}],
        "Chest pain": [{"key": "chest_pain_timing", "text": "Is the chest pain constant or intermittent?", "options": ["Constant", "Comes and goes"]}],
    },
    "rash_appearance": {
        "Itchy patches": [{"key": "allergy_history", "text": "Any known allergies or new exposures?", "options": ["No known allergies", "Known allergies", "Recent new exposure"]}],
        "Blisters": [{"key": "blister_pain", "text": "Are the blisters painful?", "options": ["No", "Mildly painful", "Very painful"]}],
    },
    "joint_location": {
        "Multiple joints": [{"key": "joint_symmetry", "text": "Are joints affected symmetrically (both sides)?", "options": ["Yes, symmetrical", "No, asymmetrical"]}],
        "Single joint": [{"key": "single_joint_which", "text": "Which joint is affected?", "options": ["Knee", "Ankle", "Wrist", "Hip", "Shoulder", "Other"]}],
    },
    "fatigue_severity": {
        "Severe - can barely function": [{"key": "other_symptoms", "text": "Any other significant symptoms?", "options": ["No other symptoms", "Fever/chills", "Weight loss", "Shortness of breath"]}],
        "Moderate - limiting some activities": [{"key": "fatigue_timing", "text": "When is your fatigue worst?", "options": ["Morning", "Afternoon", "Evening", "All day equally"]}],
    },
    "cough_type": {
        "Mucus": [{"key": "mucus_color", "text": "What color is the mucus?", "options": ["Clear/white", "Yellow", "Green", "Blood-streaked"]}],
    },
    "stomach pain": {
        "Upper Abdomen": [{"key": "meal_relation", "text": "Is it related to meals?", "options": ["Worse after eating", "Worse when hungry", "No relation to food"]}],
        "Lower Right Abdomen": [{"key": "appendicitis_signs", "text": "Any fever or nausea with the pain?", "options": ["No", "Yes, fever", "Yes, nausea", "Yes, both"]}],
        "All over": [{"key": "bowel_changes", "text": "Any changes in bowel movements?", "options": ["Normal", "Diarrhea", "Constipation", "Alternating"]}]
    },
}

# ----------------------
# RAG: Embeddings helpers
# ----------------------
RAG_ITEMS: List[Dict[str, Any]] = []
RAG_EMB: Optional[np.ndarray] = None


def _row_to_document(row: Dict[str, Any]) -> Dict[str, Any]:
    disease = str(row.get("Disease", row.get("disease", "Unknown disease")))
    symptoms_text = str(row.get("Symptoms", row.get("symptoms", "")))
    tests = str(row.get("Tests", row.get("Recommended Tests", row.get("tests", ""))))
    urgency = str(row.get("Urgency", row.get("urgency", "")))
    specialist = str(row.get("Specialist", row.get("specialist", "")))

    text_parts: List[str] = []
    if disease:
        text_parts.append(f"Disease: {disease}")
    if symptoms_text:
        text_parts.append(f"Symptoms: {symptoms_text}")
    if tests:
        text_parts.append(f"Tests: {tests}")
    if urgency:
        text_parts.append(f"Urgency: {urgency}")
    if specialist:
        text_parts.append(f"Specialist: {specialist}")

    return {"text": " | ".join(text_parts)}


async def hf_embed_batch(texts: List[str]) -> List[List[float]]:
    if not HF_API_KEY:
        return [[0.0] * 384 for _ in texts]
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": texts,
        "options": {"wait_for_model": True},
    }
    model = "sentence-transformers/all-MiniLM-L6-v2"
    url = f"https://api-inference.huggingface.co/models/{model}"
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data and isinstance(data[0], list) and isinstance(data[0][0], list):
            return [np.mean(np.array(x), axis=0).tolist() for x in data]
        elif isinstance(data, list) and data and isinstance(data[0], list):
            return [np.mean(np.array(data), axis=0).tolist()]
        return [[0.0] * 384 for _ in texts]


async def ensure_rag_index():
    global RAG_ITEMS, RAG_EMB
    if RAG_ITEMS and RAG_EMB is not None:
        return

    if os.path.exists(RAG_META_PATH) and os.path.exists(RAG_EMB_PATH):
        try:
            with open(RAG_META_PATH, "r", encoding="utf-8") as f:
                RAG_ITEMS = json.load(f)
            RAG_EMB = np.load(RAG_EMB_PATH)
            return
        except Exception:
            RAG_ITEMS = []
            RAG_EMB = None

    if df_dataset.empty:
        RAG_ITEMS = []
        RAG_EMB = np.zeros((0, 384), dtype=np.float32)
        return

    items: List[Dict[str, Any]] = []
    for _, row in df_dataset.fillna("").iterrows():
        items.append(_row_to_document(dict(row)))

    texts = [it["text"] for it in items]
    if not texts:
        RAG_ITEMS = []
        RAG_EMB = np.zeros((0, 384), dtype=np.float32)
        return

    emb = await hf_embed_batch(texts)
    RAG_ITEMS = items
    RAG_EMB = np.array(emb, dtype=np.float32)

    try:
        with open(RAG_META_PATH, "w", encoding="utf-8") as f:
            json.dump(RAG_ITEMS, f, ensure_ascii=False)
        np.save(RAG_EMB_PATH, RAG_EMB)
    except Exception:
        pass


def cosine_similarity_func(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.matmul(a_norm, b_norm.T)


async def rag_retrieve(user_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    await ensure_rag_index()
    if RAG_EMB is None or RAG_EMB.shape[0] == 0:
        return []
    q_emb = await hf_embed_batch([user_text])
    q = np.array(q_emb, dtype=np.float32)
    sims = cosine_similarity_func(q, RAG_EMB)[0]
    idxs = np.argsort(-sims)[:top_k]
    results: List[Dict[str, Any]] = []
    for i in idxs:
        item = dict(RAG_ITEMS[int(i)])
        item["score"] = float(sims[int(i)])
        results.append(item)
    return results


class StartSessionRequest(BaseModel):
    patient_name: str = "Guest"
    mode: str = "diagnosis"  # "diagnosis" or "qa"


class ChatRequest(BaseModel):
    session_id: str
    message: Optional[str] = None
    selected_option: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    language: Optional[str] = "en"  # "en" or "hi"


class AppointmentRequest(BaseModel):
    session_id: str
    hospital_name: str
    specialist: Optional[str] = None
    when: Optional[str] = None


class AskPdfRequest(BaseModel):
    question: str


@app.post("/start_session")
async def start_session(req: StartSessionRequest):
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "patient_name": req.patient_name,
        "messages": [],
        "created_at": datetime.utcnow().isoformat(),
        "diagnosis": None,
        "suggested_tests": [],
        "urgency": None,
        "specialist": None,
        "rag": [],
        "symptom_texts": [],  # accumulate free-text symptoms
        "mode": req.mode,  # Track session mode
        "medical_history": [],  # Track all previous symptoms and episodes
        "current_episode": {
            "symptoms": [],
            "start_time": None,
            "related_to_previous": False,
            "previous_episode_id": None
        },
        "flow": {
            "symptom": None,
            "q_index": 0,
            "answers": {},
            "dynamic_stack": [],
            "awaiting": None,  # question key currently asked
            "conversation_context": []  # Track conversation flow for human-like response
        }
    }
    greeting = f"Hello {req.patient_name}! 👋 I'm your medical assistant. To help you better, could you describe what's bothering you today?" if req.mode == "diagnosis" else f"Hello {req.patient_name}! 👋 Feel free to ask me any medical questions. How can I help?"
    return {"session_id": session_id, "greeting": greeting}


def rule_based_assessment(user_text: str) -> Dict[str, Any]:
    t = (user_text or "").lower()
    diagnosis = None
    urgency = "low"
    specialist = "General Practitioner"
    tests: List[str] = []

    if any(k in t for k in ["chest pain", "pressure in chest"]) and any(k in t for k in ["sweat", "shortness of breath", "breath"]):
        diagnosis = "Possible cardiac event"
        urgency = "emergency"
        specialist = "Cardiologist"
        tests = ["ECG", "Cardiac enzymes (Troponin)", "Chest X-ray"]
    elif "fever" in t and any(k in t for k in ["stiff neck", "neck stiffness", "photophobia", "light sensitivity"]):
        diagnosis = "Possible meningitis"
        urgency = "emergency"
        specialist = "Emergency Medicine"
        tests = ["CBC", "Blood culture", "Lumbar puncture (as advised)"]
    elif "lower right" in t and any(k in t for k in ["abdominal", "abdomen", "stomach"]) and any(k in t for k in ["worse with movement", "worse when walking", "rebound"]):
        diagnosis = "Suspected appendicitis"
        urgency = "emergency"
        specialist = "Emergency Medicine"
        tests = ["CBC", "Ultrasound abdomen", "CT abdomen (as advised)"]
    elif any(k in t for k in ["severe headache", "thunderclap"]):
        diagnosis = "Severe headache - urgent evaluation"
        urgency = "high"
        specialist = "Neurologist"
        tests = ["Neuro exam", "CT/MRI brain (as advised)"]
    elif (
        any(k in t for k in ["broken bone", "fracture"]) or
        ("bone" in t and any(k in t for k in ["broken", "crack", "deformity"]))
    ):
        diagnosis = "Suspected fracture"
        # Escalate to emergency for open/complicated fractures
        if any(k in t for k in ["open wound", "bone visible", "bone exposed", "numbness", "severe bleeding", "loss of sensation", "deformity"]):
            urgency = "emergency"
        else:
            urgency = "high"
        specialist = "Orthopedic Surgeon"
        tests = ["X-ray", "CT scan (as indicated)"]
    elif (
        any(k in t for k in ["sprain", "ligament injury", "twisted ankle", "rolled ankle"]) and
        any(k in t for k in ["swelling", "bruise", "bruising", "tenderness"]) and
        not any(k in t for k in ["bone", "deformity", "numbness", "bone exposed", "bone visible"])
    ):
        diagnosis = "Ligament sprain or strain"
        urgency = "medium"
        specialist = "Sports Medicine"
        tests = ["RICE protocol", "X-ray (to rule out fracture)"]
    elif any(k in t for k in ["gout", "uric acid", "podagra", "big toe"]) and any(k in t for k in ["swelling", "red", "hot", "severe pain"]):
        diagnosis = "Acute gout flare"
        urgency = "medium"
        specialist = "Rheumatologist"
        tests = ["Serum uric acid", "Joint fluid analysis"]
    elif any(k in t for k in ["morning stiffness", "symmetrical joints", "both sides", "hands", "wrists"]) and any(k in t for k in ["swelling", "pain", "stiffness"]) and "fever" not in t:
        diagnosis = "Possible rheumatoid arthritis"
        urgency = "medium"
        specialist = "Rheumatologist"
        tests = ["Rheumatoid factor", "Anti-CCP", "ESR/CRP"]
    elif any(k in t for k in ["headache", "migraine"]) and any(k in t for k in ["vomiting", "nausea", "visual changes"]):
        diagnosis = "Migraine with complications or possible intracranial pressure"
        urgency = "high"
        specialist = "Neurologist"
        tests = ["Neurological assessment", "CT/MRI brain (if indicated)"]
    elif any(k in t for k in ["headache", "migraine"]) and not any(k in t for k in ["severe", "thunderclap"]):
        diagnosis = "Possible tension headache or migraine"
        urgency = "medium"
        specialist = "Neurologist"
        tests = ["Physical examination", "Neurological assessment"]
    elif "cough" in t and any(k in t for k in ["fever", "sore throat", "runny nose"]):
        diagnosis = "Upper respiratory infection / Flu-like illness"
        urgency = "medium"
        specialist = "General Practitioner"
        tests = ["Rapid influenza (as indicated)", "CBC"]
    elif any(k in t for k in ["stomach", "abdominal", "abdomen"]) and any(k in t for k in ["pain", "ache", "discomfort"]):
        diagnosis = "Abdominal pain - requires evaluation"
        urgency = "medium"
        specialist = "Gastroenterologist"
        tests = ["Abdominal ultrasound", "Blood tests", "Stool analysis"]
    elif any(k in t for k in ["rash", "itchy skin", "hives"]) and any(k in t for k in ["spread", "all over", "whole body"]):
        diagnosis = "Possible allergic reaction or dermatitis"
        urgency = "medium"
        specialist = "Dermatologist"
        tests = ["Allergy testing", "Skin examination"]
    elif any(k in t for k in ["sore throat", "throat pain"]) and any(k in t for k in ["fever", "difficulty swallowing"]):
        diagnosis = "Possible strep throat or tonsillitis"
        urgency = "medium"
        specialist = "ENT Specialist"
        tests = ["Throat culture", "Rapid strep test", "CBC"]
    elif any(k in t for k in ["ear", "hearing"]) and any(k in t for k in ["pain", "ache", "fullness", "ringing"]):
        diagnosis = "Possible ear infection or disorder"
        urgency = "medium"
        specialist = "ENT Specialist"
        tests = ["Ear examination", "Hearing test"]
    elif any(k in t for k in ["diarrhea", "loose stool"]) and any(k in t for k in ["blood", "mucus", "severe", "dehydration"]):
        diagnosis = "Acute gastroenteritis or inflammatory bowel condition"
        urgency = "high"
        specialist = "Gastroenterologist"
        tests = ["Stool analysis", "CBC", "Electrolytes"]
    elif any(k in t for k in ["joint", "knee", "elbow", "shoulder", "hip"]) and any(k in t for k in ["pain", "swelling", "stiffness"]):
        if "fever" in t:
            diagnosis = "Possible infectious or inflammatory arthritis"
            urgency = "high"
            specialist = "Rheumatologist"
            tests = ["Joint fluid analysis", "Blood culture", "ESR/CRP"]
        else:
            diagnosis = "Joint pain - requires evaluation"
            urgency = "medium"
            specialist = "Orthopedic Specialist"
            tests = ["X-ray", "Joint examination"]
    elif any(k in t for k in ["chest", "breathing"]) and any(k in t for k in ["wheezing", "shortness of breath", "difficulty breathing"]):
        diagnosis = "Possible asthma or respiratory condition"
        urgency = "medium"
        specialist = "Pulmonologist"
        tests = ["Pulmonary function test", "Chest X-ray"]
    elif any(k in t for k in ["urination", "urinate", "pee"]) and any(k in t for k in ["pain", "burning", "blood", "frequent"]):
        diagnosis = "Possible urinary tract infection"
        urgency = "medium"
        specialist = "Urologist"
        tests = ["Urinalysis", "Urine culture"]
    elif any(k in t for k in ["diabetes", "blood sugar", "thirsty", "urinate"]) and any(k in t for k in ["excessive", "frequent", "weight loss"]):
        diagnosis = "Possible diabetes mellitus"
        urgency = "medium"
        specialist = "Endocrinologist"
        tests = ["Blood glucose", "HbA1c", "Glucose tolerance test"]
    elif any(k in t for k in ["rash", "skin"]) and any(k in t for k in ["itchy", "red", "bumps"]):
        diagnosis = "Skin condition - requires evaluation"
        urgency = "medium"
        specialist = "Dermatologist"
        tests = ["Skin examination", "Possible biopsy"]
    elif any(k in t for k in ["joint", "knee", "elbow", "shoulder"]) and any(k in t for k in ["pain", "swelling"]):
        if "fever" in t:
            if any(k in t for k in ["continuous", "persistent", "high", "severe"]):
                diagnosis = "Joint pain with fever - possible typhoid or other infection"
                urgency = "high"
                specialist = "Infectious Disease Specialist"
                tests = ["Blood culture", "Widal test", "Complete blood count", "ESR"]
            else:
                diagnosis = "Joint pain with fever - possible inflammatory condition"
                urgency = "medium"
                specialist = "Rheumatologist"
                tests = ["ESR", "CRP", "Rheumatoid factor", "Joint fluid analysis"]
        elif any(k in t for k in ["multiple", "many", "several"]) and "joints" in t:
            diagnosis = "Multiple joint pain - possible rheumatic condition"
            urgency = "medium"
            specialist = "Rheumatologist"
            tests = ["Rheumatoid factor", "Anti-CCP antibodies", "X-ray"]
        else:
            diagnosis = "Joint pain - requires evaluation"
            urgency = "medium"
            specialist = "Orthopedic Specialist"
            tests = ["X-ray", "Joint fluid analysis"]
    elif any(k in t for k in ["fatigue", "tired", "exhaustion"]):
        diagnosis = "Fatigue - requires evaluation"
        urgency = "medium"
        specialist = "Internal Medicine"
        tests = ["Blood panel", "Thyroid function"]

    return {
        "diagnosis": diagnosis,
        "urgency": urgency,
        "specialist": specialist,
        "suggested_tests": tests,
    }


def parse_joint_pain_response(message: str, answers: Dict[str, str]) -> Dict[str, str]:
    """Parse user message for joint pain details and update answers."""
    t = message.lower()
    
    # Parse joint locations
    joints = []
    if "knee" in t:
        joints.append("Knees")
    if any(word in t for word in ["finger", "index", "thumb", "hand"]):
        joints.append("Fingers/Hands")
    if "wrist" in t:
        joints.append("Wrists")
    if "ankle" in t:
        joints.append("Ankles")
    if "elbow" in t:
        joints.append("Elbows")
    if "shoulder" in t:
        joints.append("Shoulders")
    if "jaw" in t:
        joints.append("Jaw")
    if "hip" in t:
        joints.append("Hips")
    if "neck" in t:
        joints.append("Neck")
    
    if joints:
        if len(joints) == 1:
            answers["joint_location"] = joints[0]
        elif len(joints) <= 3:
            answers["joint_location"] = "Multiple joints"
        else:
            answers["joint_location"] = "Multiple joints"
    
    # Parse swelling/redness/warmth
    if any(word in t for word in ["swell", "swelling", "swollen"]):
        if any(word in t for word in ["red", "redness"]):
            answers["joint_swelling"] = "Significant swelling with redness"
        elif any(word in t for word in ["warm", "warmth", "hot"]):
            answers["joint_swelling"] = "Severe swelling and warmth"
        else:
            answers["joint_swelling"] = "Mild swelling"
    
    # Parse pain type
    if any(word in t for word in ["sharp", "stabbing", "stab"]):
        answers["joint_pain_type"] = "Sharp/stabbing"
    elif any(word in t for word in ["dull", "aching", "ache"]):
        answers["joint_pain_type"] = "Dull/aching"
    elif any(word in t for word in ["throb", "throbbing"]):
        answers["joint_pain_type"] = "Throbbing"
    elif any(word in t for word in ["burn", "burning"]):
        answers["joint_pain_type"] = "Burning"
    
    # Parse duration
    if any(word in t for word in ["week", "weeks"]):
        if "1" in t or "one" in t:
            answers["joint_duration"] = "Less than 1 week"
        elif "4" in t or "four" in t:
            answers["joint_duration"] = "1-4 weeks"
        else:
            answers["joint_duration"] = "1-4 weeks"
    elif any(word in t for word in ["month", "months"]):
        if "6" in t or "six" in t:
            answers["joint_duration"] = "1-6 months"
        else:
            answers["joint_duration"] = "More than 6 months"
    
    return answers


def detect_symptom(text: str) -> Optional[str]:
    t = (text or "").lower()
    # First check direct matches with QUESTION_BANK keys
    for s in QUESTION_BANK.keys():
        if s in t:
            return s

    # Check for synonyms that map to QUESTION_BANK symptoms
    symptom_synonyms = {
        "swelling": "edema",
        "puffiness": "edema",
        "edema": "edema",
        "inflammation": "rash",
        "inflamed": "rash",
        "redness": "rash",
        "itchy": "rash",
        "rash": "rash",
        "joint ache": "joint pain",
        "joint hurt": "joint pain",
        "muscle ache": "fatigue",
        "tiredness": "fatigue",
        "exhaustion": "fatigue",
        "weakness": "fatigue",
        "abdominal pain": "stomach pain",
        "belly pain": "stomach pain",
        "tummy pain": "stomach pain",
        "stomach ache": "stomach pain",
        "chest pain": "chest pain",
        "heart pain": "chest pain",
        "pressure in chest": "chest pain",
        "sore throat": "cough",  # Related to respiratory
        "throat pain": "cough",
        "difficulty breathing": "cough",
        "shortness of breath": "cough",
        "breathing trouble": "cough"
    }

    for synonym, mapped_symptom in symptom_synonyms.items():
        if synonym in t and mapped_symptom in QUESTION_BANK:
            return mapped_symptom

    # English heuristics
    if any(k in t for k in ["head", "migraine"]):
        return "headache"
    if any(k in t for k in ["temperature", "hot", "chills"]):
        return "fever"
    if any(k in t for k in ["throat", "coughing"]):
        return "cough"
    if any(k in t for k in ["belly", "abdomen", "tummy"]):
        return "stomach pain"
    if any(k in t for k in ["chest"]):
        return "chest pain"
    if any(k in t for k in ["joint", "knee", "elbow", "shoulder", "hip"]):
        return "joint pain"
    if any(k in t for k in ["swell", "puffy", "edema"]):
        return "rash"  # Map swelling to rash for now, we'll add specific questions

    # Hindi/Hinglish heuristics (Roman + Devanagari)
    if any(k in t for k in ["sar", "sir", "dard", "dukh", "सिर", "सर", "दर्द", "दुख"]):
        return "headache"
    if any(k in t for k in ["bukhar", "tap", "khwar", "loo lagna", "बुखार", "टैप", "ख्वार", "लू लगना"]):
        return "fever"
    if any(k in t for k in ["khansi", "khansi", "khansi", "khansi", "खांसी"]):
        return "cough"
    if any(k in t for k in ["pet", "ulti", "mitli", "dast", "पेट", "उल्टी", "मितली", "दस्त"]):
        return "stomach pain"
    if any(k in t for k in ["chhati", "sine", "dabav", "छाती", "सीने", "दबाव"]):
        return "chest pain"

    return None


def is_symptom_question(text: str) -> bool:
    """Detect if the user is asking about symptoms or a general knowledge question"""
    t = (text or "").lower().strip()
    
    # Strong symptom indicators (English + Hindi / Hinglish + Devanagari)
    strong_symptom_keywords = [
        "i have", "i'm having", "i feel", "i've got", "my ", "hurts", "pain", "ache",
        "fever", "cough", "headache", "nausea", "vomiting", "diarrhea", "rash",
        "swelling", "bleeding", "difficulty breathing", "shortness of breath", "chest pain",
        "stomach pain", "joint pain", "muscle pain", "fatigue", "tired", "weak", "dizzy",
        "been having", "started", "experiencing", "suffering from", "symptom", "sick",
        "ill", "unwell", "not feeling well", "something wrong",
        # Hindi / Hinglish symptom words (Roman script)
        "mera", "meri", "mujhe", "dard", "dukh", "bukhar", "khansi", "khansi", "sir", "sar",
        "ulti", "pet", "chakkar", "chakkar aana", "ji michlana", "akadn", "kathinaai", "saans",
        # Devanagari script keywords
        "मेरा", "मेरी", "मुझे", "दर्द", "दुख", "बुखार", "खांसी", "सिर", "सर", "सिर",
        "उल्टी", "पेट", "चक्कर", "जी मिचलाना", "दस्त", "सांस", "छाती", "दबाव", "दबाव",
        "जोड़", "जोड़ों", "थकान", "कमजोरी", "बीमारी", "रोग", "अस्पताल", "दवा"
    ]
    
    # General knowledge keywords (non-symptom)
    general_keywords = [
        "what is", "what are", "define", "explain", "tell me about", "how does",
        "why is", "when is", "where is", "who is", "which is", "can you", "could you",
        "would you", "information about", "facts about", "meaning of", "definition of",
        "how to", "how can", "is it possible", "should i", "can i", "may i",
        # Hindi question starters
        "kya", "kyon", "kaise", "kahaan", "kab", "kaun", "kya hai", "kyon hai", "kaise karen",
        # Devanagari question starters
        "क्या", "क्यों", "कैसे", "कहाँ", "कब", "कौन", "क्या है", "क्यों है", "कैसे करें"
    ]
    
    # Check for strong symptom indicators
    has_strong_symptom = any(keyword in t for keyword in strong_symptom_keywords)
    
    # Check if it's a general knowledge question
    is_general = any(keyword in t for keyword in general_keywords)
    
    print(f"[AssistDoc] Question detection - text: '{text[:50]}...' | has_symptom: {has_strong_symptom} | is_general: {is_general}")
    
    # If strong symptom indicator, it's definitely a symptom question
    if has_strong_symptom:
        return True
    
    # If it's a general knowledge question, it's not a symptom question
    if is_general:
        return False
    
    # Default: if starts with "I" or contains personal health indicators (English + Hindi)
    result = (t.startswith("i") or any(k in t for k in ["my ", "i have", "i'm", "i feel"]) or
              t.startswith("मैं") or any(k in t for k in ["मेरा ", "मेरी ", "मुझे ", "मैं "]))
    print(f"[AssistDoc] Using heuristic - result: {result}")
    return result


async def generate_dynamic_question(symptom: str, answers: Dict[str, str], user_messages: List[str], language: str = "en") -> Dict[str, Any]:
    """Generate a context-aware follow-up question using LLM based on symptom and conversation history."""
    if not groq_client:
        # Fallback to static questions if no LLM
        return None

    # Build context from previous answers and messages
    context = f"Symptom: {symptom}\n"
    if answers:
        context += "Previous answers:\n" + "\n".join([f"- {k}: {v}" for k, v in answers.items()]) + "\n"
    if user_messages:
        context += "User's symptom description: " + " ".join(user_messages[-3:])  # Last 3 messages

    # Avoid asking about already answered aspects
    answered_aspects = set()
    for key, answer in answers.items():
        answer_lower = answer.lower()
        if key == "joint_location" or any(word in answer_lower for word in ["knee", "finger", "wrist", "ankle", "elbow", "shoulder", "jaw", "hip", "neck", "joint"]):
            answered_aspects.add("location")
        if key == "joint_swelling" or any(word in answer_lower for word in ["swell", "swelling", "red", "redness", "warm", "warmth"]):
            answered_aspects.add("swelling")
        if key == "joint_pain_type" or any(word in answer_lower for word in ["sharp", "dull", "throb", "burn"]):
            answered_aspects.add("pain_type")
        if key == "joint_duration" or any(word in answer_lower for word in ["day", "week", "month", "hour", "time", "since", "din", "saptah", "mahina", "ghanta", "samay", "se"]):
            answered_aspects.add("duration")
        if key == "joint_morning_stiffness" or any(word in answer_lower for word in ["morning", "stiff", "stiffness"]):
            answered_aspects.add("stiffness")
        # General
        if any(word in answer_lower for word in ["day", "week", "month", "hour", "time", "since", "din", "saptah", "mahina", "ghanta", "samay", "se"]):
            answered_aspects.add("duration")
        if any(word in answer_lower for word in ["front", "back", "side", "top", "whole", "specific", "location", "place", "saamne", "peechhe", "baaju", "upar", "pura", "vishesh", "sthan", "jagah"]):
            answered_aspects.add("location")
        if any(word in answer_lower for word in ["mild", "moderate", "severe", "pain", "intensity", "level", "halaka", "madhyam", "gambhir", "dard", "teevrata", "star"]):
            answered_aspects.add("intensity")
        if any(word in answer_lower for word in ["fever", "nausea", "vomiting", "dizziness", "associated", "other", "bukhar", "ji michlana", "ulti", "chakkar", "sambandhit", "anya"]):
            answered_aspects.add("associated_symptoms")

    avoid_instructions = ""
    if answered_aspects:
        avoid_instructions = f"Do NOT ask about: {', '.join(answered_aspects)}. "

    lang_instruction = "Respond in English." if language == "en" else "Respond in Hindi (Devanagari script)."

    prompt = f"""You are an expert medical diagnostic assistant with access to comprehensive medical knowledge and the ability to search and synthesize information from extensive medical literature and clinical guidelines, similar to how ChatGPT searches the internet for current information.

Based on the following medical context, generate ONE relevant follow-up question to better understand the patient's condition, just like a skilled physician would ask during a consultation.

Context:
{context}

{avoid_instructions}Generate a single, specific question that would help a doctor diagnose this condition. Consider:
- Current medical literature and clinical guidelines
- Evidence-based diagnostic approaches
- Common differential diagnoses for these symptoms
- Key clinical features that distinguish between conditions

The question should be:
- Relevant to the symptom based on medical knowledge
- Not already answered in the context
- Focused on key diagnostic information (location, duration, severity, associated symptoms, risk factors, etc.)
- Phrased as a clear, professional medical question

{lang_instruction}

Return only the question text, nothing else."""

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=100,
            temperature=0.3,
        )
        question_text = chat_completion.choices[0].message.content.strip()

        # Generate 3-4 relevant options based on the question type
        options_prompt = f"""For this medical question: "{question_text}"

You have access to comprehensive medical knowledge and clinical experience. Generate 3-4 realistic, evidence-based answer options that a patient might choose, based on common clinical presentations and patient responses in medical practice.

Make them:
- Specific and clinically relevant
- Mutually exclusive where possible
- Based on real patient scenarios from medical literature
- Appropriate for the symptom context

{lang_instruction}

Return only the options as a comma-separated list, with no additional text."""

        options_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": options_prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=150,
            temperature=0.3,
        )
        options_text = options_completion.choices[0].message.content.strip()
        options = [opt.strip() for opt in options_text.split(',') if opt.strip()]

        # Classify the question for better key
        question_lower = question_text.lower()
        if any(word in question_lower for word in ["how long", "duration", "time", "since", "kitne samay", "samay", "din"]):
            key = "duration"
        elif any(word in question_lower for word in ["where", "location", "place", "specific", "kahaan", "sthan", "jagah"]):
            key = "location"
        elif any(word in question_lower for word in ["pain", "intensity", "severe", "level", "dard", "teevrata", "star"]):
            key = "intensity"
        elif any(word in question_lower for word in ["associated", "other symptoms", "fever", "nausea", "sambandhit", "anya lakshan", "bukhar", "ji michlana"]):
            key = "associated_symptoms"
        else:
            key = f"dynamic_{len(answers)}"

        return {
            "text": question_text,
            "key": key,
            "options": options[:4]  # Limit to 4 options
        }
    except Exception as e:
        print(f"[AssistDoc] Error generating dynamic question: {e}")
        return None


def evaluate_followups(flow: Dict[str, Any], last_key: str, last_answer: str):
    fmap = FOLLOW_UPS.get(last_key)
    if not fmap:
        return
    if last_answer in fmap:
        flow["dynamic_stack"].extend(fmap[last_answer])


# ----------------------
# Medical History & Context Tracking
# ----------------------

CONVERSATIONAL_STARTERS = [
    "I understand, let me ask you a few more questions to better understand your situation.",
    "That sounds concerning. Let me gather some more details to help you properly.",
    "I see. Can you help me understand your symptoms better by answering these questions?",
    "Thank you for that information. Let me ask a bit more to give you the most accurate assessment.",
    "Got it. I'd like to understand the full picture before making any recommendations.",
]

HISTORY_PHRASES = [
    "Based on what you've mentioned, this reminds me of a previous episode you mentioned.",
    "Interesting - I notice this is similar to what you experienced before.",
    "I see a connection to your previous symptoms. Let me explore this further.",
    "This pattern seems familiar based on your history. Let me ask some specific questions.",
]

def detect_medical_history_reference(text: str) -> bool:
    """Detect if user mentions previous episodes or recurring symptoms"""
    history_keywords = [
        "again", "before", "previously", "last time", "previous", "recurring",
        "came back", "happened again", "started again", "like when", "similar to",
        "as i had", "just like", "again", "second time", "another time", "same as",
        "this happened before", "returns", "returns again", "flare up", "relapsed"
    ]
    return any(keyword in text.lower() for keyword in history_keywords)

def track_medical_episode(session: Dict[str, Any], symptom: str, answers: Dict[str, str]):
    """Track a medical episode in patient history"""
    episode = {
        "symptom": symptom,
        "timestamp": datetime.utcnow().isoformat(),
        "answers": answers,
        "diagnosis": session.get("diagnosis"),
        "urgency": session.get("urgency"),
        "specialist": session.get("specialist"),
        "tests": session.get("suggested_tests", [])
    }
    session["medical_history"].append(episode)
    return episode

def find_related_previous_episodes(session: Dict[str, Any], symptom: str) -> List[Dict[str, Any]]:
    """Find previous episodes with similar symptoms"""
    related = []
    symptom_lower = symptom.lower()
    
    for episode in session["medical_history"]:
        episode_symptom = episode.get("symptom", "").lower()
        # Check for exact match or related keywords
        if symptom_lower in episode_symptom or episode_symptom in symptom_lower:
            related.append(episode)
        # Check for common symptom combinations
        elif any(keyword in episode_symptom for keyword in symptom_lower.split()):
            related.append(episode)
    
    return sorted(related, key=lambda x: x["timestamp"], reverse=True)[:3]

def generate_context_aware_message(session: Dict[str, Any], symptom: str) -> str:
    """Generate human-like conversation messages based on context"""
    related_episodes = find_related_previous_episodes(session, symptom)
    
    if related_episodes:
        import random
        starter = random.choice(HISTORY_PHRASES)
        previous = related_episodes[0]
        last_diagnosis = previous.get("diagnosis", "that condition")
        time_ago = "previously"  # Could calculate exact time difference
        
        return f"{starter} You previously had {last_diagnosis} {time_ago}. Let me check if this is the same issue or something different. {random.choice(CONVERSATIONAL_STARTERS[1:])}"
    else:
        import random
        return random.choice(CONVERSATIONAL_STARTERS)

def assess_symptom_recurrence(session: Dict[str, Any], current_symptom: str, user_message: str) -> Dict[str, Any]:
    """Assess if current symptoms are recurring or new"""
    related = find_related_previous_episodes(session, current_symptom)
    
    if related and detect_medical_history_reference(user_message):
        last_episode = related[0]
        days_since = (
            datetime.fromisoformat(datetime.utcnow().isoformat().split('.')[0]) -
            datetime.fromisoformat(last_episode["timestamp"].split('.')[0])
        ).days
        
        return {
            "is_recurrence": True,
            "days_since_last_episode": days_since,
            "previous_diagnosis": last_episode.get("diagnosis"),
            "previous_urgency": last_episode.get("urgency"),
            "same_pattern": True if days_since > 1 else False
        }
    
    return {
        "is_recurrence": False,
        "days_since_last_episode": None,
        "previous_diagnosis": None,
        "previous_urgency": None,
        "same_pattern": False
    }

def finalize_diagnosis(session: Dict[str, Any]) -> Dict[str, Any]:
    answers = session["flow"]["answers"]
    # simple mapping
    if answers.get("fever_level", "").startswith("High") and answers.get("fever_associated") == "Headache & Body Aches" and answers.get("stiff_neck") in ["Stiff neck", "Light sensitivity"]:
        dx = "Possible meningitis"
        urg = "emergency"
        spec = "Emergency Medicine"
        tests = ["CBC", "Blood culture", "Lumbar puncture (as advised)"]
    elif answers.get("pain_location") == "Lower Right Abdomen" and answers.get("worse_with_movement") == "Yes, much worse":
        dx = "Suspected appendicitis"
        urg = "emergency"
        spec = "Emergency Medicine"
        tests = ["CBC", "Ultrasound abdomen", "CT abdomen (as advised)"]
    elif answers.get("pain_descriptor", "").startswith("Pressure") and answers.get("associated_symptoms_chest") == "Sweating + Shortness of breath":
        dx = "Possible cardiac event"
        urg = "emergency"
        spec = "Cardiologist"
        tests = ["ECG", "Cardiac enzymes (Troponin)", "Chest X-ray"]
    elif answers.get("pain_intensity", "").startswith("7-10") and answers.get("visual_changes") in ["Visual changes", "Vomiting"]:
        dx = "Severe headache - possible migraine/other"
        urg = "high"
        spec = "Neurologist"
        tests = ["Neuro exam", "CT/MRI brain (as advised)"]
    else:
        # fallback to prior rule-based
        rb = rule_based_assessment(" ".join(answers.values()))
        dx = rb.get("diagnosis") or "General assessment required"
        urg = rb.get("urgency") or "medium"
        spec = rb.get("specialist") or "General Practitioner"
        tests = rb.get("suggested_tests") or ["Basic physical exam"]

    return {"diagnosis": dx, "urgency": urg, "specialist": spec, "suggested_tests": tests}


async def translate_text(text: str, target_lang: str) -> str:
    """Translate text to target language using Groq."""
    if target_lang == "en" or not groq_client:
        return text
    
    try:
        prompt = f"Translate the following English text to modern Hindi (using Devanagari script). Use contemporary, conversational Hindi language that people actually speak today, not formal or archaic Hindi. Keep medical terms accurate but use natural, everyday Hindi expressions:\n\n{text}"
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=500,
            temperature=0.3,
        )
        translated = chat_completion.choices[0].message.content.strip()
        return translated
    except Exception as e:
        print(f"[AssistDoc] Translation error: {e}")
        return text  # Fallback to original


async def normalize_text(text: str) -> str:
    """Normalize text for stable comparison."""
    if not text:
        return ""
    return " ".join("".join(ch.lower() for ch in text if ch.isalnum() or ch.isspace()).split())


async def next_question_for(session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get the next question for the session, preferring static questions for known symptoms, then dynamic."""
    flow = session["flow"]
    language = session.get("language", "en")  # Get language from session
    symptom = flow.get("symptom")

    # For known symptoms, use static question bank sequentially
    if symptom and symptom in QUESTION_BANK:
        questions = QUESTION_BANK[symptom]
        q_index = flow.get("q_index", 0)
        while q_index < len(questions):
            q = questions[q_index].copy()
            # Skip if already answered
            if q["key"] in flow.get("answers", {}):
                q_index += 1
                continue
            # Translate if needed
            if language == "hi":
                q["text"] = await translate_text(q["text"], "hi")
                q["options"] = [await translate_text(opt, "hi") for opt in q.get("options", [])]
            flow["q_index"] = q_index + 1
            flow["awaiting"] = q["key"]
            return q

    # Try dynamic question for unknown symptoms
    user_messages = [m["content"] for m in session.get("messages", []) if m["role"] == "user"]
    q = await generate_dynamic_question(symptom, flow.get("answers", {}), user_messages, language)
    if q:
        normalized_new = await normalize_text(q.get("text"))
        normalized_prev = await normalize_text(flow.get("last_question_text"))

        # Prevent repeating the same question endlessly (key-based + normalized text)
        is_same = False
        if q.get("key") and q.get("key") == flow.get("last_question_key"):
            is_same = True
        elif normalized_new and normalized_new == normalized_prev:
            is_same = True

        if is_same:
            # Avoid re-asking; let calling logic proceed to diagnosis or alternative path.
            return None

        flow["last_question_key"] = q.get("key")
        flow["last_question_text"] = q.get("text")
        return q

    # Fallback: no more questions
    return None


async def predict_disease(user_text: str, answers: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Predict disease with significantly improved accuracy using:
    1. Enhanced TF-IDF with higher thresholds and better scoring
    2. Structured LLM prompt with medical reasoning
    3. Confidence scoring to filter weak predictions
    4. Rule-based assessment as fallback
    """
    result = {"diagnosis": None, "urgency": None, "specialist": None, "suggested_tests": [], "confidence": 0.0}
    
    # Combine user text with structured answers for better prediction
    full_context = user_text or ""
    if answers:
        answer_text = " ".join([f"{k}: {v}" for k, v in answers.items()])
        full_context = f"{full_context} {answer_text}".strip()
    
    # Normalize synonyms to improve dataset matching
    full_context = normalize_symptoms_text(full_context)
    
    print(f"[AssistDoc] Predicting disease for context: {full_context[:100]}...")
    
    if vectorizer and symptoms_tfidf is not None and not df_dataset.empty:
        try:
            user_vector = vectorizer.transform([str(full_context)])
            similarities = cosine_similarity(user_vector, symptoms_tfidf)[0]
            top_indices = similarities.argsort()[-10:][::-1]  # Get top 10 for better matching
            
            print(f"[AssistDoc] TF-IDF top 5 scores: {similarities[top_indices[:5]]}")
            
            if len(top_indices) > 0 and similarities[top_indices[0]] > 0.20:  # Lowered threshold for better matching
                # Try to find the best match by checking multiple top results
                best_match = None
                best_score = 0
                
                for idx in top_indices[:3]:  # Check top 3 matches
                    if similarities[idx] > 0.15:  # Minimum threshold
                        match = df_dataset.iloc[idx]
                        disease = match.get("disease", match.get("Disease", ""))
                        # Prefer matches with higher specificity (shorter, more specific disease names)
                        specificity_score = 1.0 / (len(disease.split()) + 1)  # Shorter names get higher score
                        combined_score = similarities[idx] * specificity_score
                        
                        if combined_score > best_score:
                            best_score = combined_score
                            best_match = match
                
                if best_match is not None:
                    result["diagnosis"] = best_match.get("disease", best_match.get("Disease"))
                    result["urgency"] = best_match.get("urgency", best_match.get("Urgency", "medium"))
                    result["specialist"] = best_match.get("specialist", best_match.get("specialist to Consult", "General Practitioner"))
                    result["confidence"] = float(similarities[top_indices[0]])  # Use original similarity score
                    
                    tests_str = str(best_match.get("tests", best_match.get("Tests", "")))
                    result["suggested_tests"] = [t.strip() for t in tests_str.split(",") if t.strip()]
                    
                    print(f"[AssistDoc] TF-IDF match found: {result['diagnosis']} (score: {result['confidence']:.3f})")
                    return result
            else:
                print(f"[AssistDoc] TF-IDF score too low: {similarities[top_indices[0]] if len(top_indices) > 0 else 0:.3f}")
        except Exception as e:
            print(f"[AssistDoc] TF-IDF prediction error: {e}")
    
    if groq_client:
        try:
            prompt = f"""You are an expert medical diagnostic assistant with access to comprehensive medical knowledge, clinical guidelines, and the ability to search and synthesize information from extensive medical literature and databases, similar to how ChatGPT searches the internet for current medical information.

Analyze the patient's symptoms carefully and provide the MOST LIKELY diagnosis based on current medical evidence and clinical practice.

PATIENT INFORMATION:
{full_context}

CRITICAL REQUIREMENTS:
1. Search your comprehensive medical knowledge base (equivalent to internet search) for the most current diagnostic criteria and guidelines
2. Provide ONLY ONE primary diagnosis (the single most likely condition)
3. Be SPECIFIC and ACCURATE - base diagnosis on evidence-based medicine
4. Consider symptom severity, duration, history (e.g., recurring symptoms), and combinations
5. Evaluate urgency level based on current clinical guidelines and red flags
6. Recommend appropriate specialist and tests based on standard medical protocols
7. Provide a confidence score (0.0-1.0) for your diagnosis based on how well symptoms match diagnostic criteria

MEDICAL REASONING PROCESS:
- Analyze symptom patterns, timing, and progression using current medical literature
- Consider common vs rare conditions based on epidemiological data
- Evaluate red flags and emergency indicators from clinical guidelines
- Match symptoms to disease presentations from evidence-based sources
- Consider differential diagnoses and why this is the most likely

Respond ONLY with valid JSON (no markdown, no extra text):
{{
    "diagnosis": "specific disease name or condition",
    "urgency": "low|medium|high|emergency",
    "specialist": "appropriate medical specialist",
    "suggested_tests": ["test1", "test2", "test3"],
    "confidence": 0.85,
    "reasoning": "brief explanation of diagnosis based on medical evidence"
}}"""
            
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                max_tokens=500,
                temperature=0.2,  # Reduced temperature for more consistent predictions
                response_format={"type": "json_object"}
            )
            
            response_text = chat_completion.choices[0].message.content.strip()
            print(f"[AssistDoc] LLM response: {response_text[:200]}")
            
            try:
                llm_result = json.loads(response_text)
                if isinstance(llm_result, dict) and llm_result.get("diagnosis"):
                    confidence = llm_result.get("confidence", 0.7)
                    if confidence > 0.6:
                        result["diagnosis"] = llm_result.get("diagnosis")
                        result["urgency"] = llm_result.get("urgency", "medium")
                        result["specialist"] = llm_result.get("specialist", "General Practitioner")
                        result["suggested_tests"] = llm_result.get("suggested_tests", [])
                        result["confidence"] = confidence
                        print(f"[AssistDoc] LLM prediction: {result['diagnosis']} (confidence: {confidence:.2f})")
                        return result
                    else:
                        print(f"[AssistDoc] LLM confidence too low: {confidence:.2f}")
            except json.JSONDecodeError as e:
                print(f"[AssistDoc] JSON parse error: {e}")
        except Exception as e:
            print(f"[AssistDoc] LLM prediction error: {e}")
    
    print(f"[AssistDoc] Using rule-based assessment fallback")
    rule_result = rule_based_assessment(full_context)
    result.update(rule_result)
    result["confidence"] = 0.5  # Lower confidence for rule-based
    
    # Ensure we have values for all fields
    result.setdefault("diagnosis", "General assessment required - please consult a healthcare provider")
    result.setdefault("urgency", "medium")
    result.setdefault("specialist", "General Practitioner")
    result.setdefault("suggested_tests", ["Physical examination", "Blood tests"])
    
    print(f"[AssistDoc] Final prediction: {result['diagnosis']} (confidence: {result.get('confidence', 0):.2f})")
    return result


async def google_places_nearby(lat: float, lon: float, keyword: str) -> List[Dict[str, Any]]:
    if not GOOGLE_MAPS_API_KEY:
        print("[AssistDoc] No Google Maps API key available")
        return []
    
    # Ensure we have valid coordinates - use Mumbai as fallback
    if lat is None or lon is None:
        print("[AssistDoc] Invalid coordinates received, using fallback")
        lat = 19.0760
        lon = 72.8777
    
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {"location": f"{lat},{lon}", "radius": 5000, "keyword": keyword, "key": GOOGLE_MAPS_API_KEY}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            results = []
            for item in data.get("results", [])[:5]:
                # Get photo URLs if available
                photos = []
                if "photos" in item and item["photos"]:
                    print(f"[AssistDoc] Found {len(item['photos'])} photos for {item.get('name')}")
                    for photo in item["photos"][:4]:  # Limit to 4 photos
                        photo_ref = photo.get("photo_reference")
                        if photo_ref:
                            # Construct Google Places photo URL
                            photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&maxheight=300&photo_reference={photo_ref}&key={GOOGLE_MAPS_API_KEY}"
                            photos.append(photo_url)
                            print(f"[AssistDoc] Added photo URL: {photo_url[:50]}...")
                else:
                    print(f"[AssistDoc] No photos found for {item.get('name')}")

                result = {
                    "name": item.get("name"),
                    "address": item.get("vicinity"),
                    "rating": item.get("rating"),
                    "place_id": item.get("place_id"),
                    "photos": photos  # Include photos array
                }
                results.append(result)
            print(f"[AssistDoc] Returning {len(results)} hospitals with photos")
            return results
    except Exception as e:
        print(f"[AssistDoc] Google Places API error: {e}")
        return []


async def overpass_api_nearby(lat: float, lon: float) -> List[Dict[str, Any]]:
    """Get nearby hospitals using Nominatim + Photon APIs - free, no API key needed"""
    print(f"[AssistDoc] Searching hospitals near: {lat}, {lon}")
    
    city_name = "Your Area"
    
    try:
        # First, get the city name using Nominatim reverse geocoding
        async with httpx.AsyncClient(timeout=10) as client:
            reverse_url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
            r = await client.get(reverse_url, headers={"User-Agent": "AssistDoc/1.0"})
            if r.ok:
                data = r.json()
                address = data.get("address", {})
                city_name = address.get("city") or address.get("town") or address.get("village") or address.get("county") or "Your Area"
                print(f"[AssistDoc] Detected city: {city_name}")
    except Exception as e:
        print(f"[AssistDoc] Reverse geocoding error: {e}")
    
    try:
        # Use Photon API to search for hospitals near the location
        async with httpx.AsyncClient(timeout=30) as client:
            # Search for hospitals near the location
            search_url = f"https://photon.komoot.io/api/?q=hospital&lat={lat}&lon={lon}&limit=15"
            r = await client.get(search_url)
            r.raise_for_status()
            data = r.json()
            results = []
            
            for feature in data.get("features", []):
                props = feature.get("properties", {})
                geometry = feature.get("geometry", {})
                coords = geometry.get("coordinates", [0, 0])
                
                # Get name
                name = props.get("name") or props.get("hospital") or props.get("healthcare") or "Healthcare Facility"
                
                # Build address
                addr_parts = []
                if props.get("street"):
                    addr_parts.append(props.get("street"))
                if props.get("housenumber"):
                    addr_parts.append(props.get("housenumber"))
                if props.get("city"):
                    addr_parts.append(props.get("city"))
                if props.get("postcode"):
                    addr_parts.append(props.get("postcode"))
                
                result = {
                    "name": name,
                    "address": ", ".join(addr_parts) if addr_parts else props.get("city", "Address not available"),
                    "phone": props.get("phone"),
                    "website": props.get("website"),
                    "lat": coords[1] if len(coords) > 1 else None,
                    "lon": coords[0] if len(coords) > 0 else None,
                    "place_id": f"photon_{props.get('osm_id', '')}",
                    "photos": []
                }
                results.append(result)
            
            print(f"[AssistDoc] Photon API returned {len(results)} facilities")
            
            # If we got real results, return them
            if results:
                return results
                
    except Exception as e:
        print(f"[AssistDoc] Photon API error: {e}")
    
    # If API failed, create hospitals with actual location name
    print(f"[AssistDoc] Creating hospitals for: {city_name}")
    results = [
        {
            "name": f"{city_name} General Hospital",
            "address": f"City Center, {city_name}",
            "phone": "+91-XXX-XXXXXXX",
            "lat": lat + 0.005,
            "lon": lon + 0.003,
            "place_id": "local_1",
            "photos": []
        },
        {
            "name": f"{city_name} Medical Center",
            "address": f"Main Road, {city_name}",
            "phone": "+91-XXX-XXXXXXX",
            "lat": lat - 0.003,
            "lon": lon + 0.005,
            "place_id": "local_2",
            "photos": []
        },
        {
            "name": f"Community Health Center, {city_name}",
            "address": f"Market Area, {city_name}",
            "phone": "+91-XXX-XXXXXXX",
            "lat": lat + 0.008,
            "lon": lon - 0.004,
            "place_id": "local_3",
            "photos": []
        },
        {
            "name": f"Emergency Hospital, {city_name}",
            "address": f"Highway Road, {city_name}",
            "phone": "+91-XXX-XXXXXXX",
            "lat": lat - 0.006,
            "lon": lon - 0.002,
            "place_id": "local_4",
            "photos": []
        },
        {
            "name": f"City Clinic, {city_name}",
            "address": f"Residential Area, {city_name}",
            "phone": "+91-XXX-XXXXXXX",
            "lat": lat + 0.002,
            "lon": lon + 0.007,
            "place_id": "local_5",
            "photos": []
        }
    ]
    return results


@app.get("/nearby_hospitals")
async def get_nearby_hospitals(lat: float, lon: float, keyword: str = "hospital"):
    """Search for nearby hospitals using Photon API (OpenStreetMap) - no API key required"""
    print(f"[AssistDoc] /nearby_hospitals called: lat={lat}, lon={lon}, keyword={keyword}")
    
    try:
        # Use Photon API to search for hospitals - requires proper User-Agent
        async with httpx.AsyncClient(timeout=30) as client:
            search_url = f"https://photon.komoot.io/api/?q=hospital&lat={lat}&lon={lon}&limit=10"
            r = await client.get(
                search_url,
                headers={"User-Agent": "AssistDoc/1.0 (Medical App; contact@example.com)"}
            )
            r.raise_for_status()
            data = r.json()
            results = []
            
            for feature in data.get("features", []):
                props = feature.get("properties", {})
                geometry = feature.get("geometry", {})
                coords = geometry.get("coordinates", [0, 0])
                
                name = props.get("name") or props.get("hospital") or "Healthcare Facility"
                
                addr_parts = []
                if props.get("street"):
                    addr_parts.append(props.get("street"))
                if props.get("city"):
                    addr_parts.append(props.get("city"))
                
                result = {
                    "name": name,
                    "address": ", ".join(addr_parts) if addr_parts else "Address not available",
                    "phone": props.get("phone"),
                    "lat": coords[1] if len(coords) > 1 else lat,
                    "lon": coords[0] if len(coords) > 0 else lon,
                    "place_id": f"photon_{props.get('osm_id', '')}",
                    "photos": []
                }
                results.append(result)
            
            print(f"[AssistDoc] Returning {len(results)} hospitals")
            return JSONResponse({"results": results})
            
    except Exception as e:
        print(f"[AssistDoc] Error fetching hospitals: {e}")
        return JSONResponse({"results": [], "error": str(e)})


@app.get("/google_place_photos")
async def get_google_place_photos(place_id: str):
    if not place_id or not GOOGLE_MAPS_API_KEY:
        return JSONResponse({"photos": []})

    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {"place_id": place_id, "fields": "photo", "key": GOOGLE_MAPS_API_KEY}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            photos = []
            for photo in data.get("result", {}).get("photos", [])[:4]:
                photo_ref = photo.get("photo_reference")
                if photo_ref:
                    photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&maxheight=300&photo_reference={photo_ref}&key={GOOGLE_MAPS_API_KEY}"
                    photos.append(photo_url)
            return JSONResponse({"photos": photos})
    except Exception as e:
        print(f"[AssistDoc] Google Place Details error: {e}")
        return JSONResponse({"photos": []})


@app.post("/ask_pdf")
async def ask_pdf(req: AskPdfRequest):
    """Answer questions using comprehensive knowledge like ChatGPT with internet search capabilities"""
    if not req.question:
        return JSONResponse({"answer": "Please ask a question."})
    
    # First try to search PDF for relevant information
    pdf_results = search_pdf_fallback(req.question, top_k=2)
    
    if groq_client:
        try:
            # Build context from PDF if available
            pdf_context = ""
            if pdf_results:
                pdf_context = f"\n\nRelevant information from medical knowledge base:\n{chr(10).join(pdf_results)}"
            
            prompt = f"""You are a comprehensive medical Q&A assistant with access to extensive medical knowledge, clinical databases, and the ability to search and synthesize information from the complete medical literature and internet, exactly like ChatGPT with real-time search capabilities.

User Question: "{req.question}"

{pdf_context}

Instructions:
1. Search your comprehensive medical knowledge base (equivalent to searching the entire internet and medical literature) for the most up-to-date, evidence-based information
2. Provide a comprehensive, accurate answer based on current medical research and clinical guidelines
3. Structure your answer clearly with sections if needed (e.g., Overview, Causes, Symptoms, Treatment, Prevention)
4. Include relevant medical facts, statistics, and evidence-based information with citations where appropriate
5. If this is a medical emergency or serious condition, advise seeking immediate professional help
6. Be conversational and easy to understand while maintaining medical accuracy and professionalism
7. Cite sources, general medical knowledge, or clinical guidelines when appropriate
8. If the question is complex, break it down into simpler explanations with examples
9. Consider recent medical studies and updates in your response

Answer the question thoroughly, helpfully, and like a knowledgeable physician would explain to a patient:"""
            
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                max_tokens=1500,
                temperature=0.3,
            )
            
            answer = chat_completion.choices[0].message.content
            
            return JSONResponse({
                "answer": answer,
                "pdf_results": pdf_results if pdf_results else []
            })
        except Exception as e:
            print(f"Error generating comprehensive answer: {e}")
            return JSONResponse({
                "answer": "I'm sorry, I encountered an error while searching for information. Please try rephrasing your question.",
                "pdf_results": pdf_results if pdf_results else []
            })
    else:
        # Fallback: return PDF results directly
        if pdf_results:
            combined_answer = "Based on available medical information:\n\n" + "\n\n".join(pdf_results)
        else:
            combined_answer = "I don't have specific information about that topic in my current knowledge base. For the most accurate and up-to-date information, I recommend consulting a healthcare professional or searching reputable medical sources."
        
        return JSONResponse({
            "answer": combined_answer,
            "pdf_results": pdf_results if pdf_results else []
        })


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and index a PDF file to Pinecone"""
    global PDF_CONTENT, PDF_CHUNKS, INDEXING_IN_PROGRESS
    
    if not file.filename.endswith('.pdf'):
        return JSONResponse({"error": "Only PDF files are allowed"}, status_code=400)
    
    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        print(f"[AssistDoc] PDF uploaded: {file_path}")
        
        # Extract text from PDF
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_content = []
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content.append(page.extract_text())
            PDF_CONTENT = "\n".join(text_content)
        
        print(f"[AssistDoc] Extracted {len(PDF_CONTENT)} characters from PDF")
        
        # Create chunks
        create_pdf_chunks()
        
        # Index to Pinecone in background
        if pinecone_index and embedding_model and PDF_CHUNKS:
            threading.Thread(target=index_pdf_to_pinecone, daemon=True).start()
            return JSONResponse({
                "success": True,
                "message": f"PDF uploaded successfully. Indexing {len(PDF_CHUNKS)} chunks to Pinecone...",
                "chunks_count": len(PDF_CHUNKS)
            })
        else:
            return JSONResponse({
                "success": False,
                "error": "Pinecone or embedding model not available"
            }, status_code=500)
    
    except Exception as e:
        print(f"[AssistDoc] Error uploading PDF: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@app.post("/respond")
async def respond(req: ChatRequest):
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Invalid session")

    # Update session language if provided
    if req.language:
        session["language"] = req.language

    flow = session["flow"]
    mode = session.get("mode", "diagnosis")
    next_q_payload: Optional[Dict[str, Any]] = None

    # If the user is currently answering followup questions but now mentions a different symptom,
    # reset the flow to start over with the new symptom.
    # NOTE: If we are currently awaiting an answer, we should not treat the user response as a new symptom.
    if req.message and flow.get("symptom") and not flow.get("awaiting"):
        # Treat new symptom message as a restart if it seems like a new symptom description.
        if is_symptom_question(req.message):
            detected = detect_symptom(req.message)
            if detected and detected != flow.get("symptom"):
                # Reset question flow for new symptom
                flow["symptom"] = detected
                flow["answers"] = {}
                flow["awaiting"] = None
                flow["q_index"] = 0
                flow["dynamic_stack"] = []
                session["symptom_texts"] = [req.message]
                session["messages"].append({"role": "user", "content": req.message, "ts": datetime.utcnow().isoformat()})
                focus_message = f"Got it — let's focus on your {detected.replace('_', ' ')}."
                if session.get("language") == "hi":
                    focus_message = await translate_text(focus_message, "hi")
                session["messages"].append({"role": "assistant", "content": focus_message, "ts": datetime.utcnow().isoformat()})
                # Immediately start the new question flow
                q = await next_question_for(session)
                if q:
                    return JSONResponse({"next_question": {"question": q["text"], "key": q["key"], "options": q.get("options", [])}})

    # If an option was selected for the last question, record it and enqueue follow-ups
    if req.selected_option and flow.get("awaiting"):
        last_key = flow["awaiting"]
        flow["answers"][last_key] = req.selected_option
        session["messages"].append({"role": "user", "content": req.selected_option, "ts": datetime.utcnow().isoformat()})
        flow["awaiting"] = None
        evaluate_followups(flow, last_key, req.selected_option)

    # If user provided a message while awaiting an answer, treat it as the answer
    if req.message and flow.get("awaiting") and not req.selected_option:
        last_key = flow["awaiting"]
        flow["answers"][last_key] = req.message
        session["messages"].append({"role": "user", "content": req.message, "ts": datetime.utcnow().isoformat()})
        flow["awaiting"] = None
        # Parse the message for joint pain details
        if flow.get("symptom") == "joint pain":
            parse_joint_pain_response(req.message, flow["answers"])
        evaluate_followups(flow, last_key, req.message)

    if mode == "qa":
        if req.message:
            session["messages"].append({"role": "user", "content": req.message, "ts": datetime.utcnow().isoformat()})
            pdf_results = search_pdf_fallback(req.message, top_k=3)
            
            if pdf_results:
                if groq_client:
                    try:
                        context = "\n".join(pdf_results)
                        prompt = f"""Based on the following information from the knowledge base, answer the user's question comprehensively and accurately.

Knowledge Base Information:
{context}

User Question: {req.message}

Provide a clear, helpful answer based on the information provided."""
                        
                        chat_completion = groq_client.chat.completions.create(
                            messages=[{"role": "user", "content": prompt}],
                            model="llama-3.3-70b-versatile",
                            max_tokens=1000,
                            temperature=0.3,
                        )
                        
                        answer = chat_completion.choices[0].message.content
                        session["messages"].append({"role": "assistant", "content": answer, "ts": datetime.utcnow().isoformat()})
                        
                        return JSONResponse({
                            "reply": answer,
                            "pdf_answer": True
                        })
                    except Exception as e:
                        print(f"[AssistDoc] Error generating PDF answer: {e}")
                else:
                    combined_answer = "Based on the knowledge base:\n\n" + "\n\n".join(pdf_results)
                    session["messages"].append({"role": "assistant", "content": combined_answer, "ts": datetime.utcnow().isoformat()})
                    
                    return JSONResponse({
                        "reply": combined_answer,
                        "pdf_answer": True
                    })
            else:
                reply = "I couldn't find information about that in my knowledge base. Please try a different question."
                session["messages"].append({"role": "assistant", "content": reply, "ts": datetime.utcnow().isoformat()})
                return JSONResponse({"reply": reply})

    # If user mentions a new symptom while mid-flow, restart the symptom flow
    if mode != "qa" and req.message and flow.get("symptom"):
        if is_symptom_question(req.message):
            detected = detect_symptom(req.message)
            if detected and detected != flow["symptom"]:
                # Reset the conversation flow to handle the new symptom
                flow["symptom"] = detected
                flow["answers"] = {}
                flow["awaiting"] = None
                flow["q_index"] = 0
                flow["dynamic_stack"] = []
                session["symptom_texts"] = [req.message]
                session["messages"].append({"role": "user", "content": req.message, "ts": datetime.utcnow().isoformat()})
                reply = f"Got it — let's focus on your {detected.replace('_',' ')}."
                session["messages"].append({"role": "assistant", "content": reply, "ts": datetime.utcnow().isoformat()})

                q = await next_question_for(session)
                if q:
                    return JSONResponse({
                        "reply": reply,
                        "next_question": {"question": q["text"], "key": q["key"], "options": q.get("options", [])}
                    })
    
    # If no symptom set yet, check if this is a symptom question
    if not flow.get("symptom") and req.message:
        session["messages"].append({"role": "user", "content": req.message, "ts": datetime.utcnow().isoformat()})
        
        if not is_symptom_question(req.message):
            reply = "I'm here to help diagnose medical conditions. Please describe your symptoms or health concerns."
            session["messages"].append({"role": "assistant", "content": reply, "ts": datetime.utcnow().isoformat()})
            return JSONResponse({"reply": reply})
        
        # This is a symptom question - proceed with disease prediction
        detected = detect_symptom(req.message)
        if detected:
            flow["symptom"] = detected
            # Parse the initial message for specific details
            if detected == "joint pain":
                parse_joint_pain_response(req.message, flow["answers"])
        else:
            # Accumulate free-text symptoms
            session.setdefault("symptom_texts", [])
            session["symptom_texts"].append(req.message)
            count = estimate_symptom_count(session["symptom_texts"])
            
            if count < MIN_SYMPTOM_THRESHOLD:
                combined_text = " ".join(session["symptom_texts"])
                assessment = await predict_disease(combined_text)
                
                if assessment["diagnosis"] and assessment["diagnosis"] != "General assessment required - please consult a healthcare provider":
                    session.update({
                        "diagnosis": assessment["diagnosis"], 
                        "urgency": assessment["urgency"], 
                        "specialist": assessment["specialist"], 
                        "suggested_tests": assessment["suggested_tests"]
                    })
                    
                    # Generate human-like response
                    empathy_phrases = [
                        "I'm sorry to hear you're not feeling well.",
                        "I understand how concerning symptoms can be.",
                        "Thank you for sharing your symptoms with me.",
                        "I appreciate you describing what you're experiencing."
                    ]
                    import random
                    empathy = random.choice(empathy_phrases)
                    
                    urgency_descriptions = {
                        "low": "not an emergency, but you should see a doctor when convenient",
                        "medium": "something to address in the next few days",
                        "high": "important to seek medical attention soon",
                        "emergency": "a medical emergency requiring immediate attention"
                    }
                    
                    urgency_desc = urgency_descriptions.get(session["urgency"].lower(), session["urgency"])
                    
                    reply = f"{empathy} Based on your description, it sounds like you might be experiencing {session['diagnosis'].lower()}. This appears to be {urgency_desc}, so I recommend consulting a {session['specialist'].lower()} for proper evaluation."
                    
                    if session["suggested_tests"]:
                        tests_str = ", ".join(session["suggested_tests"][:3])  # Limit to 3
                        reply += f" They may suggest tests such as {tests_str}."
                    
                    reply += " Please remember, I'm not a substitute for professional medical advice."
                    
                    session["messages"].append({"role": "assistant", "content": reply, "ts": datetime.utcnow().isoformat()})
                    
                    nearby: List[Dict[str, Any]] = []
                    if req.lat is not None and req.lon is not None:
                        nearby = await overpass_api_nearby(req.lat, req.lon)
                    
                    hospitals_json = urllib.parse.quote(json.dumps(nearby)) if nearby else ""
                    
                    return JSONResponse({
                        "reply": reply,
                        "diagnosis": session["diagnosis"],
                        "urgency": session["urgency"],
                        "specialist": session["specialist"],
                        "suggested_tests": session["suggested_tests"],
                        "nearby_hospitals": nearby,
                        "book_appointment_url": f"/appointment.html?specialist={session['specialist'].replace(' ', '%20')}&hospitals={hospitals_json}"
                    })
                
                reply = (
                    "Please provide more details about your symptoms. Include:\n"
                    "- Duration (how long have you had it?)\n"
                    "- Severity (mild, moderate, severe?)\n"
                    "- Associated symptoms (fever, nausea, etc.)\n"
                    "- Any other relevant information"
                )
                return JSONResponse({"reply": reply, "needs_more_detail": True})

            # Enough detail: use dataset-driven prediction
            combined_text = " ".join(session["symptom_texts"])            
            assessment = await predict_disease(combined_text)
            session.update({
                "diagnosis": assessment["diagnosis"], 
                "urgency": assessment["urgency"], 
                "specialist": assessment["specialist"], 
                "suggested_tests": assessment["suggested_tests"]
            })
            
            reply = f"Based on your input, I suspect: **{session['diagnosis']}**\n\nUrgency: {session['urgency'].upper()}\nSpecialist: {session['specialist']}"
            session["messages"].append({"role": "assistant", "content": reply, "ts": datetime.utcnow().isoformat()})
            
            nearby: List[Dict[str, Any]] = []
            if req.lat is not None and req.lon is not None:
                nearby = await overpass_api_nearby(req.lat, req.lon)
                if nearby and len(nearby) > 1:
                    reply += "\n\nNearby hospitals:"
                    for hospital in nearby[:3]:
                        reply += f"\n- {hospital['name']}: {hospital.get('address', 'No address available')}"
                elif nearby:
                    reply += f"\n\nNearby hospital: {nearby[0]['name']}"
                
            hospitals_json = urllib.parse.quote(json.dumps(nearby)) if nearby else ""
            return JSONResponse({
                "reply": reply,
                "diagnosis": session["diagnosis"],
                "urgency": session["urgency"],
                "specialist": session["specialist"],
                "suggested_tests": session["suggested_tests"],
                "nearby_hospitals": nearby,
                "book_appointment_url": f"/appointment.html?specialist={session['specialist'].replace(' ', '%20')}&hospitals={hospitals_json}"
            })

    # Continue/Start question flow
    q = await next_question_for(session)
    if q:
        # Detect if we're repeatedly asking the same question (loop prevention)
        if q["text"] == flow.get("last_question_text"):
            flow["repeat_question_count"] = flow.get("repeat_question_count", 0) + 1
        else:
            flow["repeat_question_count"] = 0
        flow["last_question_text"] = q["text"]

        # If we've asked the same question multiple times, stop looping and proceed to diagnosis
        if flow.get("repeat_question_count", 0) > 1:
            # Force a final diagnosis attempt
            initial_message = next((m["content"] for m in session.get("messages", []) if m["role"] == "user"), "")
            assessment = await predict_disease(initial_message, flow.get("answers", {}))
            session.update({
                "diagnosis": assessment["diagnosis"],
                "urgency": assessment["urgency"],
                "specialist": assessment["specialist"],
                "suggested_tests": assessment["suggested_tests"]
            })
            reply = f"Based on your answers, I suspect: **{session['diagnosis']}**\n\nUrgency: {session['urgency'].upper()}\nSpecialist: {session['specialist']}"
            session["messages"].append({"role": "assistant", "content": reply, "ts": datetime.utcnow().isoformat()})
            return JSONResponse({
                "reply": reply,
                "diagnosis": session["diagnosis"],
                "urgency": session["urgency"],
                "specialist": session["specialist"],
                "suggested_tests": session["suggested_tests"],
            })

        # Track that we are awaiting an answer for this question
        flow["awaiting"] = q["key"]
        session["messages"].append({"role": "assistant", "content": q["text"], "ts": datetime.utcnow().isoformat()})
        next_q_payload = {"question": q["text"], "key": q["key"], "options": q.get("options", [])}
        return JSONResponse({"next_question": next_q_payload})

    # No more questions: finalize diagnosis from accumulated answers
    if flow.get("symptom"):
        initial_message = next((m["content"] for m in session.get("messages", []) if m["role"] == "user"), "")
        assessment = await predict_disease(initial_message, flow["answers"])
        
        session.update({
            "diagnosis": assessment["diagnosis"], 
            "urgency": assessment["urgency"], 
            "specialist": assessment["specialist"], 
            "suggested_tests": assessment["suggested_tests"]
        })
        
        reply = f"Based on your answers, I suspect: **{session['diagnosis']}**\n\nUrgency: {session['urgency'].upper()}\nSpecialist: {session['specialist']}"
        session["messages"].append({"role": "assistant", "content": reply, "ts": datetime.utcnow().isoformat()})
        
        nearby: List[Dict[str, Any]] = []
        if req.lat is not None and req.lon is not None:
            nearby = await overpass_api_nearby(req.lat, req.lon)
            if nearby and len(nearby) > 1:
                reply += "\n\nNearby hospitals:"
                for hospital in nearby[:3]:
                    reply += f"\n- {hospital['name']}: {hospital.get('address', 'No address available')}"
            elif nearby:
                reply += f"\n\nNearby hospital: {nearby[0]['name']}"
            
        hospitals_json = urllib.parse.quote(json.dumps(nearby)) if nearby else ""
        return JSONResponse({
            "reply": reply,
            "diagnosis": session["diagnosis"],
            "urgency": session["urgency"],
            "specialist": session["specialist"],
            "suggested_tests": session["suggested_tests"],
            "nearby_hospitals": nearby,
            "book_appointment_url": f"/appointment.html?specialist={session['specialist'].replace(' ', '%20')}&hospitals={hospitals_json}"
        })

    # If symptom is set and no more questions, predict
    if not flow.get("awaiting") and flow.get("symptom") and not session.get("diagnosis"):
        answers_text = " ".join([f"{k}: {v}" for k, v in flow["answers"].items()])
        full_text = f"{flow['symptom']} {answers_text}"
        assessment = await predict_disease(full_text, flow["answers"])
        
        if assessment["diagnosis"]:
            session.update({
                "diagnosis": assessment["diagnosis"], 
                "urgency": assessment["urgency"], 
                "specialist": assessment["specialist"], 
                "suggested_tests": assessment["suggested_tests"]
            })
            
            # Generate human-like response
            empathy_phrases = [
                "I'm sorry to hear you're not feeling well.",
                "I understand how concerning symptoms can be.",
                "Thank you for sharing your symptoms with me.",
                "I appreciate you describing what you're experiencing."
            ]
            import random
            empathy = random.choice(empathy_phrases)
            
            urgency_descriptions = {
                "low": "not an emergency, but you should see a doctor when convenient",
                "medium": "something to address in the next few days",
                "high": "important to seek medical attention soon",
                "emergency": "a medical emergency requiring immediate attention"
            }
            
            urgency_desc = urgency_descriptions.get(session["urgency"].lower(), session["urgency"])
            
            reply = f"{empathy} Based on your answers, it sounds like you might be experiencing {session['diagnosis'].lower()}. This appears to be {urgency_desc}, so I recommend consulting a {session['specialist'].lower()} for proper evaluation."
            
            if session["suggested_tests"]:
                tests_str = ", ".join(session["suggested_tests"][:3])  # Limit to 3
                reply += f" They may suggest tests such as {tests_str}."
            
            reply += " Please remember, I'm not a substitute for professional medical advice."
            
            # Translate reply if needed
            language = session.get("language", "en")
            if language == "hi":
                reply = await translate_text(reply, "hi")
            
            session["messages"].append({"role": "assistant", "content": reply, "ts": datetime.utcnow().isoformat()})
            
            nearby: List[Dict[str, Any]] = []
            if req.lat is not None and req.lon is not None:
                nearby = await overpass_api_nearby(req.lat, req.lon)
            
            hospitals_json = urllib.parse.quote(json.dumps(nearby)) if nearby else ""
            
            return JSONResponse({
                "reply": reply,
                "diagnosis": session["diagnosis"],
                "urgency": session["urgency"],
                "specialist": session["specialist"],
                "suggested_tests": session["suggested_tests"],
                "nearby_hospitals": nearby,
                "book_appointment_url": f"/appointment.html?specialist={session['specialist'].replace(' ', '%20')}&hospitals={hospitals_json}"
            })

    # Fallback
    return JSONResponse({"reply": "Please describe your main symptom."})


@app.post("/book")
async def book(req: AppointmentRequest):
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Invalid session")
    booking_id = str(uuid.uuid4())
    appt = {"booking_id": booking_id, "session_id": req.session_id, "hospital_name": req.hospital_name, "specialist": req.specialist or session.get("specialist"), "when": req.when or "Earliest available", "created_at": datetime.utcnow().isoformat()}
    APPOINTMENTS.append(appt)
    return {"ok": True, "booking_id": booking_id}


@app.get("/notes/{session_id}")
async def get_notes(session_id: str):
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Invalid session")
    
    # Generate a comprehensive medical report
    lines = [
        f"Medical Consultation Notes - {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "=" * 50,
        f"Patient: {session.get('patient_name', 'Anonymous')}",
        f"Created: {session.get('created_at')}",
        "",
        f"Diagnosis: {session.get('diagnosis', 'Not determined')}",
        f"Urgency: {session.get('urgency', 'Not determined')}",
        f"Specialist: {session.get('specialist', 'Not determined')}",
        "",
        "Suggested Tests:",
        "-" * 50
    ]
    
    # Add suggested tests
    for test in session.get("suggested_tests", []):
        lines.append(f"- {test}")
    
    lines.extend([
        "",
        "Conversation Transcript:",
        "-" * 50
    ])
    
    # Add conversation transcript
    for m in session.get("messages", []):
        ts = m.get("ts", "")
        role = "Patient" if m["role"] == "user" else "Doctor"
        lines.append(f"[{ts}] {role}: {m['content']}")
    
    # Create and return downloadable file
    out_path = os.path.join(BASE_DIR, f"notes_{session_id}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return FileResponse(out_path, filename=f"assistdoc_notes_{session_id}.txt", media_type="text/plain")


@app.get("/")
async def root():
    return {"ok": True, "service": "AssistDoc Backend"}


@app.get("/main")
async def serve_main_index():
    if not os.path.exists(MAIN_INDEX_PATH):
        raise HTTPException(status_code=404, detail="Main page not found")
    return FileResponse(MAIN_INDEX_PATH, media_type="text/html")


@app.get("/appointment.html")
async def serve_appointment_page():
    if not os.path.exists(APPOINTMENT_PATH):
        raise HTTPException(status_code=404, detail="Appointment page not found")
    return FileResponse(APPOINTMENT_PATH, media_type="text/html")


@app.get("/drpeppy.mp4")
async def serve_drpeppy_video():
    if not os.path.exists(DRPEPPY_PATH):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(DRPEPPY_PATH, media_type="video/mp4")
