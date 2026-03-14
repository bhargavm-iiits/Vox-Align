import chromadb
import pyttsx3
import warnings
warnings.filterwarnings('ignore')

print("🚀 Initializing VoxAlign Memory & Mouth Backend...")

# ==========================================
# 🧠 1. THE MEMORY (ChromaDB Vector Database)
# ==========================================
# Create a local, temporary database in your computer's RAM
chroma_client = chromadb.Client()
memory_collection = chroma_client.create_collection(name="voxcorp_knowledge_base")

# 📚 Feed the AI its "Rulebook" (This is the RAG Knowledge Base!)
print("📚 Loading VoxCorp Rulebook into Vector Database...")
memory_collection.add(
    documents=[
        "COMPANY POLICY 1: If the user is ANGRY about a broken product, apologize immediately and offer them a full refund. Assure them our engineering team is fixing the bug.",
        "COMPANY POLICY 2: If the user is SAD or feeling down, be extremely gentle. Offer them our 'Cheer Up' 20% discount code: SMILE20.",
        "COMPANY POLICY 3: If the user is HAPPY, match their energy! Congratulate them and ask if they would be willing to leave a 5-star review on our website."
    ],
    metadatas=[{"topic": "angry_user"}, {"topic": "sad_user"}, {"topic": "happy_user"}],
    ids=["policy_angry", "policy_sad", "policy_happy"]
)

# ==========================================
# 👄 2. THE MOUTH (pyttsx3 Text-to-Speech)
# ==========================================
tts_engine = pyttsx3.init()
# Optional: Change the voice (0 is usually male, 1 is usually female on Windows)
voices = tts_engine.getProperty('voices')
if len(voices) > 1:
    tts_engine.setProperty('voice', voices[1].id)
# Slow down the speaking rate slightly for a more empathetic tone
tts_engine.setProperty('rate', 160) 

def speak_out_loud(text):
    print(f"\n🗣️ AI is speaking: '{text}'")
    tts_engine.say(text)
    tts_engine.runAndWait()

# ==========================================
# 🤖 3. THE LOGIC ENGINE (Mock LLM)
# ==========================================
def generate_empathetic_response(user_text, detected_emotion):
    print(f"\n🔍 Searching Memory for how to handle a {detected_emotion} user...")
    
    # 1. RAG Retrieval: Search the database using the user's text AND emotion
    search_query = f"User is {detected_emotion}. They said: {user_text}"
    results = memory_collection.query(
        query_texts=[search_query],
        n_results=1 # Just get the #1 most relevant policy
    )
    
    retrieved_policy = results['documents'][0][0]
    print(f"📄 Memory Retrieved: {retrieved_policy}")
    
    # 2. Generate the Response (In the future, a real LLM like Llama-3 does this part!)
    # For right now, we will construct a smart, automated response based on the RAG rules:
    if "ANGRY" in detected_emotion:
        final_response = "I am so incredibly sorry you are dealing with this. I hear your frustration. As per our policy, I am processing a full refund for you right now, and our engineers have been notified."
    elif "SAD" in detected_emotion:
        final_response = "I am so sorry you are having a hard time. Please be gentle with yourself today. If it helps, I can offer you our special 20 percent discount code: SMILE 20."
    elif "HAPPY" in detected_emotion:
        final_response = "That is absolutely fantastic news! I am so thrilled for you. Since you are having such a great experience, would you mind leaving us a five-star review?"
    else:
        final_response = f"I understand. You seem to be feeling {detected_emotion}. How else can I assist you today?"
        
    return final_response

# ==========================================
# 🧪 4. MANUAL TESTING LOOP
# ==========================================
if __name__ == "__main__":
    print("\n✅ Backend Memory & Mouth Online!")
    print("Let's simulate the output from your VoxAlign Perception script.")
    
    while True:
        print("\n" + "="*50)
        test_text = input("⌨️  Type a fake user prompt (or 'q' to quit): ")
        if test_text.lower() == 'q':
            break
            
        test_emotion = input("🎭 Enter the detected emotion (Angry, Sad, Happy, Neutral): ").upper()
        
        # Run the RAG generation!
        ai_response = generate_empathetic_response(user_text=test_text, detected_emotion=test_emotion)
        
        # Speak it out loud!
        speak_out_loud(ai_response)