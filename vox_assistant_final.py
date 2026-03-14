import os
import torch
import torch.nn as nn
import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper
import warnings
import chromadb
import pandas as pd
import re
import tempfile
import threading
import tkinter as tk
from tkinter import simpledialog
from concurrent.futures import ThreadPoolExecutor
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
from google import genai
import logging
import asyncio
import edge_tts
import pygame

# ==========================================
# ⚙️ 1. PATHS & SETTINGS
# ==========================================
BASE_DIR = r"D:\Projects\VoxAlign"
CSV_PATH = os.path.join(BASE_DIR, r"Preprocessing\Preprocessig_outputs\customers.csv")
WAV2VEC_WEIGHTS = os.path.join(BASE_DIR, r"Preprocessing\Preprocessig_outputs\VoxAlign_Wav2Vec_Enhanced_Calibrated.pth")
BERT_WEIGHTS = os.path.join(BASE_DIR, r"Preprocessing\Preprocessig_outputs\VoxAlign_TextBrain_BERT.pth")
DB_PATH = "./voxalign_db"

# GEMINI_API_KEY = "AIzaSyBJbfRa8-1r68PDoP6zEbmRVclLj2rEEv4"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_gemini_key_here") # type: ignore
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTIONS = {0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Angry', 4: 'Fear', 5: 'Disgust', 6: 'Surprise'}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

pygame.mixer.init()

# ==========================================
# 🧠 2. AI CORE MODELS
# ==========================================
client = genai.Client(api_key=GEMINI_API_KEY) # type: ignore
executor = ThreadPoolExecutor(max_workers=4)
whisper_model = whisper.load_model("tiny").to(DEVICE)
w2v_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class VoxAlignWav2Vec(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.classifier = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 7))
    def forward(self, x):
        outputs = self.wav2vec2(x)
        pooled = torch.mean(outputs.last_hidden_state, dim=1)
        return self.classifier(pooled)

audio_model = VoxAlignWav2Vec().to(DEVICE)
audio_model.load_state_dict(torch.load(WAV2VEC_WEIGHTS, map_location=DEVICE)['model_state_dict'])
audio_model.eval()

text_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=7).to(DEVICE) # type: ignore
text_model.load_state_dict(torch.load(BERT_WEIGHTS, map_location=DEVICE))
text_model.eval()

chroma_client = chromadb.PersistentClient(path=DB_PATH)
memory_collection = chroma_client.get_or_create_collection(name="saas_production_db")

# ==========================================
# 🎙️ 3. ROBUST TTS ENGINE (Synchronous)
# ==========================================
def speak_out_loud(text, wait=False):
    """
    Generates and plays audio. 
    Pauses the program until audio finishes so the mic doesn't hear it.
    """
    logging.info(f"🗣️ AI Response: {text}")
    temp_audio = tempfile.mktemp(suffix=".mp3")
    try:
        communicate = edge_tts.Communicate(text, "en-US-ChristopherNeural", rate="-15%")
        asyncio.run(communicate.save(temp_audio))
        
        pygame.mixer.music.load(temp_audio)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
    except Exception as e:
        logging.error(f"TTS Error: {e}")
    finally:
        pygame.mixer.music.unload()
        if os.path.exists(temp_audio):
            try: os.remove(temp_audio)
            except: pass

# ==========================================
# 🎤 4. FAST INPUT RECORDING
# ==========================================
import queue
def record_audio(fs=16000, silence_limit=1.2): # 🔥 FIX: Reduced silence limit to 1.2s for faster response
    q = queue.Queue()
    def callback(indata, frames, time, status): q.put(indata.copy())
    
    logging.info("🔴 LISTENING...")
    frames, silent_chunks, started = [], 0, False
    with sd.InputStream(samplerate=fs, channels=1, dtype='int16', callback=callback, blocksize=1600):
        while True:
            data = q.get()
            frames.append(data)
            rms = np.sqrt(np.mean(np.square(data.astype(np.float32))))
            
            if rms > 1500: 
                started, silent_chunks = True, 0
            elif started:
                silent_chunks += 1
            
            if started and silent_chunks > int(silence_limit / 0.1): break
            if len(frames) > 300: break 
            
    path = tempfile.mktemp(suffix=".wav")
    sf.write(path, np.concatenate(frames), fs)
    return path

# ==========================================
# 🧠 5. REQUEST PROCESSING (DYNAMIC)
# ==========================================
def process_request(customer_name, transcription, audio_path, order_id):
    def get_emot():
        with torch.no_grad():
            inputs = bert_tokenizer(transcription, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            t_probs = F.softmax(text_model(**inputs).logits, dim=1)
            
            speech, _ = sf.read(audio_path)
            if len(speech.shape) > 1: speech = speech[:, 0]
            speech = np.pad(speech, (0, max(0, 64000 - len(speech))), 'constant')[:64000]
            
            a_inputs = w2v_extractor(speech, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
            a_probs = F.softmax(audio_model(a_inputs), dim=1)
            return EMOTIONS[torch.argmax((a_probs + t_probs) / 2.0).item()].upper() # type: ignore

    emotion = executor.submit(get_emot).result()
    rag_results = memory_collection.query(query_texts=[transcription], n_results=1)
    policy = rag_results['documents'][0][0] if rag_results['documents'] else "Provide general assistance."

    clean_text = transcription.replace('"', '').replace("'", "").strip()
    
    # 🔥 FIX: Dynamic Prompting. Tells the AI to switch modes based on the user's question.
    prompt = f"""
    You are a highly intelligent, professional AI Assistant speaking with {customer_name}. 
    Their current emotion is detected as: {emotion}.
    
    User Message: "{clean_text}"
    Company Database Context: "{policy}"

    INSTRUCTIONS:
    1. If the user asks about an order, tracking, refunds, or company policies, use the 'Company Database Context' to answer them as a Customer Support Agent.
    2. If the user asks a general knowledge, coding, or technology question (e.g., "how does AI work?"), ignore the company context and answer their question directly and comprehensively like a standard AI.
    3. Keep your response conversational and spoken-word friendly.
    4. Always end your response by asking: "Is there anything else I can help you with?"
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt
        )
        speak_out_loud(response.text.strip()) # type: ignore
    except Exception as e:
        logging.error(f"Gemini API Error: {e}")
        speak_out_loud("I am experiencing a connection issue. Please repeat that.")

# ==========================================
# 🚀 6. VERIFICATION & MAIN LOOP
# ==========================================
def verify_customer():
    if not os.path.exists(CSV_PATH): return None
    df = pd.read_csv(CSV_PATH).dropna(subset=['order_id', 'name'])
    
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    
    speak_out_loud("Identity verification required. Please check the pop-up window and enter your OTP.")
    oid = simpledialog.askstring("Security", "Enter OTP:", parent=root)
    root.destroy()
    
    if oid:
        df['order_id_str'] = df['order_id'].astype(float).astype(int).astype(str)
        match = df[df['order_id_str'] == str(oid).strip()]
        if not match.empty: 
            return match.iloc[0]['name'].strip(), oid
    return None

if __name__ == "__main__":
    print("\n" + "="*40 + "\n✅ VOXALIGN OMNI-ENGINE: ONLINE (GEMINI)\n" + "="*40)
    auth = verify_customer()
    
    if auth:
        cust_name, cust_id = auth
        speak_out_loud(f"Hello {cust_name}. How can I help you today?")
        
        while True:
            # 🔥 FIX: Removed time.sleep(2.0). The script now listens instantly after it finishes speaking.
            audio_file = record_audio()
            result = whisper_model.transcribe(audio_file)
            text = result["text"].strip() # type: ignore
            
            if text:
                if len(text) > 4: 
                    print(f"User: {text}")
                    if re.search(r"\b(exit|quit|bye|goodbye|thanks|thank you|no thanks|no thank you)\b", text.lower()):
                        speak_out_loud("You are very welcome. Have a great day. Goodbye!", wait=True)
                        break
                    process_request(cust_name, text, audio_file, cust_id)
            
            if os.path.exists(audio_file): os.remove(audio_file)
    else:
        speak_out_loud("Access Denied.", wait=True)