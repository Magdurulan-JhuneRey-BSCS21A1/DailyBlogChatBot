import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
with open("data.json", "r") as f:
    faq_data = json.load(f)

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]

# Load sentence transformer model
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode all questions once at startup
print("Encoding questions...")
question_embeddings = model.encode(questions, convert_to_tensor=False)

def get_response(user_input):
    # Encode user input
    user_embedding = model.encode([user_input], convert_to_tensor=False)
    
    # Calculate similarities
    similarities = cosine_similarity(user_embedding, question_embeddings)
    index = similarities.argmax()
    score = similarities[0][index]
    
    # Lower threshold since semantic similarity is more reliable
    if score < 0.5:
        return "I'm not sure about that. Can you rephrase or ask something else?"
    
    return answers[index]

# Chat loop
if __name__ == "__main__":
    print("\nChatbot ready! Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        print("Bot:", get_response(user_input))