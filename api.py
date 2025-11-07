from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import re
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import numpy as np

# Download NLTK data
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

app = Flask(__name__)
CORS(app)  # allow React or any frontend

# -----------------------------
# Initialize NLP components
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# Helper Functions
# -----------------------------
def preprocess_basic(text):
    """Basic preprocessing without keyword filtering"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

def preprocess(text, important_keywords):
    """Enhanced preprocessing with important keyword preservation"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = nltk.word_tokenize(text)
    
    # Keep important keywords and non-stopwords
    tokens = [
        lemmatizer.lemmatize(w) for w in tokens 
        if w not in stop_words or w in important_keywords
    ]
    
    return " ".join(tokens)

def expand_synonyms_selective(text, important_keywords, max_synonyms=2):
    """
    Selective synonym expansion - only expand non-specific words
    Avoids expanding domain-specific terms like 'post', 'blog', 'login'
    """
    words = text.split()
    expanded = list(words)
    
    for w in words:
        # Skip expansion for important domain keywords
        if w in important_keywords:
            continue
            
        synonym_count = 0
        for syn in wordnet.synsets(w):
            if synonym_count >= max_synonyms:
                break
            for lemma in syn.lemmas()[:2]:
                if synonym_count >= max_synonyms:
                    break
                synonym = lemma.name().replace("_", " ").lower()
                if synonym != w and synonym not in expanded:
                    expanded.append(synonym)
                    synonym_count += 1
    
    return " ".join(expanded)

def calculate_keyword_overlap(user_tokens, faq_tokens, important_keywords):
    """Calculate overlap with emphasis on important keywords"""
    user_set = set(user_tokens.split())
    faq_set = set(faq_tokens.split())
    
    if not user_set:
        return 0
    
    # Regular overlap
    overlap = len(user_set.intersection(faq_set))
    regular_score = overlap / len(user_set)
    
    # Important keyword overlap (weighted higher)
    important_user = user_set.intersection(important_keywords)
    important_faq = faq_set.intersection(important_keywords)
    
    if important_user:
        important_overlap = len(important_user.intersection(important_faq))
        important_score = important_overlap / len(important_user)
        # Blend regular and important keyword scores
        return 0.4 * regular_score + 0.6 * important_score
    
    return regular_score

def calculate_sequence_similarity(str1, str2):
    """Calculate similarity based on sequence matching"""
    return SequenceMatcher(None, str1, str2).ratio()

def detect_intent_keywords(user_input, important_keywords):
    """
    Detect key intent words in user query
    Returns a score based on presence of important keywords
    """
    words = set(preprocess_basic(user_input).split())
    intent_words = words.intersection(important_keywords)
    
    if not words:
        return 0
    
    return len(intent_words) / len(words)

# -----------------------------
# Load FAQ dataset & Initialize
# -----------------------------
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract important keywords from data.json
important_keywords_list = data.get("important_keywords", [])
all_important_keywords = set(important_keywords_list)

# Get FAQ items
faq_data = data.get("faqs", data if isinstance(data, list) else [])

# Store processed versions
questions_expanded = [
    expand_synonyms_selective(preprocess(item["question"], all_important_keywords), 
                             all_important_keywords, max_synonyms=1)
    for item in faq_data
]
questions_original = [
    preprocess(item["question"], all_important_keywords) 
    for item in faq_data
]
answers = [item["answer"] for item in faq_data]

# TF-IDF vectorizers
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=5000,
    min_df=1,
    sublinear_tf=True,
    token_pattern=r'\b\w+\b'
)
X_expanded = vectorizer.fit_transform(questions_expanded)

vectorizer_strict = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=3000,
    min_df=1,
    token_pattern=r'\b\w+\b'
)
X_strict = vectorizer_strict.fit_transform(questions_original)

# -----------------------------
# Main Response Function
# -----------------------------
def get_response(user_input, confidence_threshold=0.30):
    """
    Enhanced response matching with multiple scoring methods
    
    Args:
        user_input: User's question
        confidence_threshold: Minimum confidence score (0-1)
    """
    # Preprocess user input
    processed_input = preprocess(user_input, all_important_keywords)
    processed_expanded = expand_synonyms_selective(
        processed_input, 
        all_important_keywords, 
        max_synonyms=1
    )
    
    # Check query length and adjust threshold
    query_length = len(processed_input.split())
    
    # Very short queries need higher confidence
    if query_length <= 1:
        confidence_threshold = max(confidence_threshold, 0.5)
    elif query_length == 2:
        confidence_threshold = max(confidence_threshold, 0.4)
    
    # Method 1: TF-IDF with controlled synonym expansion
    user_vec_expanded = vectorizer.transform([processed_expanded])
    sim_expanded = cosine_similarity(user_vec_expanded, X_expanded)[0]
    
    # Method 2: TF-IDF strict (no synonyms)
    user_vec_strict = vectorizer_strict.transform([processed_input])
    sim_strict = cosine_similarity(user_vec_strict, X_strict)[0]
    
    # Method 3: Keyword overlap with important keyword emphasis
    keyword_scores = np.array([
        calculate_keyword_overlap(processed_input, q, all_important_keywords) 
        for q in questions_original
    ])
    
    # Method 4: Intent keyword detection
    intent_score = detect_intent_keywords(user_input, all_important_keywords)
    intent_boost = 1.0 + (intent_score * 0.3)  # Up to 30% boost for strong intent
    
    # Method 5: Sequence similarity
    sequence_scores = np.array([
        calculate_sequence_similarity(processed_input, q)
        for q in questions_original
    ])
    
    # Combine scores with weighted average
    combined_scores = (
        0.20 * sim_expanded +
        0.35 * sim_strict +
        0.30 * keyword_scores +
        0.15 * sequence_scores
    ) * intent_boost
    
    # Get best match
    best_idx = combined_scores.argmax()
    best_score = combined_scores[best_idx]
    
    # Check confidence threshold
    if best_score < confidence_threshold:
        # Check if there are multiple close matches (ambiguous)
        second_best_score = np.partition(combined_scores, -2)[-2]
        if best_score > 0.2 and (best_score - second_best_score) < 0.08:
            return "I found multiple possible answers. Could you be more specific or rephrase your question?"
        
        # Provide helpful fallback
        if intent_score > 0.3:
            return "I understand you're asking about that topic, but I need more details. Can you rephrase your question?"
        
        return "I'm not sure about that. Can you rephrase or provide more details?"
    
    return answers[best_idx]

# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "message": "DailyBlogs Support Chatbot API",
        "endpoints": {
            "/chat": "POST - Send a message to the chatbot"
        }
    })

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        
        if not data or "message" not in data:
            return jsonify({"error": "No message provided"}), 400
        
        message = data.get("message", "").strip()
        
        if not message:
            return jsonify({"error": "Empty message"}), 400
        
        # Get response from enhanced chatbot
        reply = get_response(message)
        
        return jsonify({
            "reply": reply,
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({
            "error": "An error occurred processing your request",
            "details": str(e)
        }), 500

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("="*60)
    print("  DAILYBLOGS SUPPORT CHATBOT API")
    print("="*60)
    print(f"\n✓ Server starting on http://0.0.0.0:{port}")
    print(f"✓ Loaded {len(faq_data)} FAQ entries")
    print(f"✓ Tracking {len(all_important_keywords)} important keywords")
    print("\n" + "-"*60 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False)
