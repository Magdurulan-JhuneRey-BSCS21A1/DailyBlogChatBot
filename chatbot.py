import json
import re
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data if not already done
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def expand_synonyms(text):
    words = text.split()
    expanded = set(words)
    for w in words:
        for syn in wordnet.synsets(w):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace("_", " "))
    return " ".join(expanded)

# Load FAQ data
with open("data.json", "r") as f:
    faq_data = json.load(f)

questions = [expand_synonyms(preprocess(item["question"])) for item in faq_data]
answers = [item["answer"] for item in faq_data]

# TF-IDF with n-grams
vectorizer = TfidfVectorizer(ngram_range=(1, 3))
X = vectorizer.fit_transform(questions)

def get_response(user_input):
    processed_input = expand_synonyms(preprocess(user_input))
    user_vec = vectorizer.transform([processed_input])
    similarities = cosine_similarity(user_vec, X)
    index = similarities.argmax()
    score = similarities[0][index]

    if score < 0.25:  # Slightly lower threshold for flexibility
        return "I'm not sure about that. Can you rephrase?"
    return answers[index]

# Chat loop
if __name__ == "__main__":
    print("Chatbot ready! Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        print("Bot:", get_response(user_input))
