from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # allow React or any frontend

# -----------------------------
# Load FAQ dataset
# -----------------------------
with open("data.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]

# Prepare TF-IDF model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# -----------------------------
# Function to get best response
# -----------------------------
def get_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, X)
    index = similarities.argmax()
    score = similarities[0][index]

    if score < 0.3:
        return "🤔 I'm not sure about that. Could you rephrase?"
    return answers[index]

# -----------------------------
# Flask route
# -----------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    reply = get_response(message)
    return jsonify({"reply": reply})

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
