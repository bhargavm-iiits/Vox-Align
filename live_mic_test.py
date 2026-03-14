import torch
import torch.nn as nn
import librosa
import whisper
import numpy as np
import warnings
import sounddevice as sd
from scipy.io.wavfile import write
import os
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

warnings.filterwarnings('ignore')

# --- 1. SETTINGS & PATHS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTIONS = {0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Angry', 4: 'Fear', 5: 'Disgust', 6: 'Surprise'}

WAV2VEC_WEIGHTS = r"D:\Projects\VoxAlign\Preprocessing\Preprocessig_outputs\VoxAlign_Wav2Vec_Enhanced_Calibrated.pth"
BERT_WEIGHTS = r"D:\Projects\VoxAlign\Preprocessing\Preprocessig_outputs\VoxAlign_TextBrain_BERT.pth"

print(f"🚀 Loading VoxAlign Live Multimodal Engine on {DEVICE}...")

# --- 2. LOAD WHISPER (The Bridge) ---
print("Loading Whisper...")
whisper_model = whisper.load_model("base").to(DEVICE)

# --- 3. LOAD AUDIO BRAIN (Wav2Vec 2.0) ---
print("Loading Audio Brain...")
w2v_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

class VoxAlignWav2Vec(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 7)
        )

    def forward(self, x):
        outputs = self.wav2vec2(x)
        pooled = torch.mean(outputs.last_hidden_state, dim=1)
        return self.classifier(pooled)

audio_model = VoxAlignWav2Vec().to(DEVICE)
checkpoint = torch.load(WAV2VEC_WEIGHTS, map_location=DEVICE)
audio_model.load_state_dict(checkpoint['model_state_dict'])
TEMPERATURE = checkpoint.get('temperature', 1.0)
audio_model.eval()

# --- 4. LOAD TEXT BRAIN (BERT) ---
print("Loading Text Brain...")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=7).to(DEVICE)
text_model.load_state_dict(torch.load(BERT_WEIGHTS, map_location=DEVICE))
text_model.eval()

# --- 5. MICROPHONE RECORDING FUNCTION ---
def record_audio(duration=5, fs=16000, filename="temp_mic.wav"):
    print(f"\n🔴 RECORDING FOR {duration} SECONDS...")
    print("Speak now!")
    
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait() 
    
    print("✅ Recording complete.")
    write(filename, fs, myrecording)
    return filename

# --- 6. INFERENCE PIPELINE ---
def analyze_live_audio(audio_path, manual_text=""):
    
    # --- Step A: Transcription (Whisper OR Manual Override) ---
    if manual_text:
        transcription = manual_text
        print(f"\n📝 User typed manually: '{transcription}'")
    else:
        print("\n📝 Listening with Whisper...")
        transcription = whisper_model.transcribe(audio_path)["text"].strip()
        print(f"📝 Whisper heard: '{transcription}'")
    
    if not transcription:
        print("⚠️ No text detected or entered. Try again!")
        return

    with torch.no_grad():
        # --- Step B: Text Brain Prediction ---
        inputs = bert_tokenizer(transcription, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
        text_logits = text_model(**inputs).logits
        text_probs = F.softmax(text_logits, dim=1)
        
        # --- Step C: Audio Brain Prediction ---
        speech, _ = librosa.load(audio_path, sr=16000)
        max_length = 16000 * 4
        if len(speech) > max_length:
            speech = speech[:max_length]
        else:
            speech = np.pad(speech, (0, max_length - len(speech)), 'constant')
        
        audio_inputs = w2v_extractor(speech, sampling_rate=16000, return_tensors="pt",
                                      padding="max_length", max_length=max_length, truncation=True).input_values.to(DEVICE)
        
        audio_logits = audio_model(audio_inputs)
        audio_probs = F.softmax(audio_logits / TEMPERATURE, dim=1)
        
        # --- Step D: Late Fusion Combiner ---
        combined_probs = (audio_probs + text_probs) / 2.0
        
        final_pred_id = torch.argmax(combined_probs, dim=1).item()
        confidence = combined_probs[0][final_pred_id].item() * 100
        
        print("\n--- 📊 AI DIAGNOSTICS ---")
        print(f"Audio Brain thought: {EMOTIONS[torch.argmax(audio_probs).item()]}")
        print(f"Text Brain thought:  {EMOTIONS[torch.argmax(text_probs).item()]}")
        print(f"✅ FINAL VOXALIGN VERDICT: {EMOTIONS[final_pred_id].upper()} ({confidence:.1f}% confidence)")

if __name__ == "__main__":
    print("\n✅ System Online!")
    
    while True:
        user_input = input("\n🎙️ Press ENTER to start recording (or type 'q' to quit): ")
        if user_input.lower() == 'q':
            print("Shutting down VoxAlign...")
            break
            
        # 1. Record the audio
        temp_file = record_audio(duration=5)
        
        # 2. Ask the user if they want to type the text manually
        print("\n⌨️  Do you want to override Whisper?")
        manual_override = input("   Type your text here (or just press ENTER to let Whisper transcribe): ").strip()
        
        # 3. Run the analysis!
        analyze_live_audio(temp_file, manual_text=manual_override)
        
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)