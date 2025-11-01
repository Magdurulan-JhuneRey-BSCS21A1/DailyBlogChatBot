import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
with open("data.json", "r") as f:
    faq_data = json.load(f)

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]

# Convert text into vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

def get_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, X)
    index = similarities.argmax()
    score = similarities[0][index]

    if score < 0.3:
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
