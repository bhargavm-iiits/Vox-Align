import torch
import torch.nn as nn
import librosa
import whisper
import numpy as np
import warnings
import os
from pathlib import Path
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
from tabulate import tabulate   # pip install tabulate if not installed

warnings.filterwarnings('ignore')

# ================== CONFIGURATION ==================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTIONS = {0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Angry', 4: 'Fear', 5: 'Disgust', 6: 'Surprise'}

# Paths to trained weights (update these to your actual paths)
WAV2VEC_WEIGHTS = r"D:\Projects\VoxAlign\Preprocessing\Preprocessig_outputs\VoxAlign_Wav2Vec_Enhanced_Calibrated.pth"
BERT_WEIGHTS = r"D:\Projects\VoxAlign\Preprocessing\Preprocessig_outputs\VoxAlign_TextBrain_BERT.pth"

# ================== LOAD MODELS ==================
print(f"🚀 Loading VoxAlign Multimodal Engine on {DEVICE}...")

# ----- Whisper (The Bridge) -----
print("Loading Whisper...")
whisper_model = whisper.load_model("base").to(DEVICE)

# ----- Audio Brain (Wav2Vec2) -----
print("Loading Audio Brain...")
w2v_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

class VoxAlignWav2Vec(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        # Freeze feature extractor (optional, but matches training)
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
        # Classifier (same architecture as during training)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        outputs = self.wav2vec2(x)
        pooled = torch.mean(outputs.last_hidden_state, dim=1)
        return self.classifier(pooled)

audio_model = VoxAlignWav2Vec().to(DEVICE)
checkpoint = torch.load(WAV2VEC_WEIGHTS, map_location=DEVICE)
audio_model.load_state_dict(checkpoint['model_state_dict'])
TEMPERATURE = checkpoint.get('temperature', 1.0)   # fallback to 1.0 if not present
print(f"🌡️ Loaded Audio Brain with Calibrated Temperature: {TEMPERATURE:.4f}")
audio_model.eval()

# ----- Text Brain (BERT) -----
print("Loading Text Brain...")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=7).to(DEVICE)
text_model.load_state_dict(torch.load(BERT_WEIGHTS, map_location=DEVICE))
text_model.eval()

print("\n✅ Models loaded successfully.\n")

# ================== ANALYSIS FUNCTION ==================
def analyze_emotion(audio_path):
    """Run multimodal emotion analysis on a single audio file and return results."""
    try:
        # Step 1: Transcribe with Whisper
        transcription = whisper_model.transcribe(audio_path)["text"].strip()

        with torch.no_grad():
            # Step 2: Text Brain prediction
            inputs = bert_tokenizer(transcription, return_tensors="pt", padding=True,
                                    truncation=True, max_length=128).to(DEVICE)
            text_logits = text_model(**inputs).logits
            text_probs = F.softmax(text_logits, dim=1)

            # Step 3: Audio Brain prediction
            speech, _ = librosa.load(audio_path, sr=16000)
            max_len = 16000 * 4  # 4 seconds (must match training)
            if len(speech) > max_len:
                speech = speech[:max_len]
            else:
                speech = np.pad(speech, (0, max_len - len(speech)), 'constant')

            audio_inputs = w2v_extractor(speech, sampling_rate=16000, return_tensors="pt",
                                         padding="max_length", max_length=max_len, truncation=True).input_values
            audio_inputs = audio_inputs.to(DEVICE)

            audio_logits = audio_model(audio_inputs)
            audio_probs = F.softmax(audio_logits / TEMPERATURE, dim=1)

            # Step 4: Late Fusion (average probabilities)
            combined_probs = (audio_probs + text_probs) / 2.0
            final_pred_id = torch.argmax(combined_probs, dim=1).item()
            confidence = combined_probs[0][final_pred_id].item() * 100

        return {
            'file': Path(audio_path).name,
            'path': audio_path,
            'transcription': transcription,
            'audio_emotion': EMOTIONS[torch.argmax(audio_probs).item()],
            'text_emotion': EMOTIONS[torch.argmax(text_probs).item()],
            'final_emotion': EMOTIONS[final_pred_id],
            'confidence': confidence
        }
    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")
        return None

# ================== FILE COLLECTION ==================
def collect_audio_files():
    """
    Define how you want to collect the 20 audio files.
    Option A: Manual list (replace with your actual file paths)
    Option B: Auto-discover from dataset folders (uncomment and adjust paths)
    """
    # ---------- OPTION A: MANUAL LIST ----------
    # Replace these with the actual paths to 20 audio files covering all emotions
    file_list = [
        r"D:\Projects\VoxAlign\Datasets\TESS\TESS Toronto emotional speech set data\YAF_neutral\YAF_chat_neutral.wav",
        r"D:\Projects\VoxAlign\Datasets\TESS\...\YAF_angry.wav",          # replace with real angry file
        r"D:\Projects\VoxAlign\Datasets\TESS\...\YAF_disgust.wav",
        r"D:\Projects\VoxAlign\Datasets\TESS\...\YAF_fear.wav",
        r"D:\Projects\VoxAlign\Datasets\TESS\...\YAF_happy.wav",
        r"D:\Projects\VoxAlign\Datasets\TESS\...\YAF_sad.wav",
        r"D:\Projects\VoxAlign\Datasets\TESS\...\YAF_surprise.wav",
        r"D:\Projects\VoxAlign\Datasets\RAVDESS\Actor_01\03-01-01-01-01-01-01.wav",   # neutral
        r"D:\Projects\VoxAlign\Datasets\RAVDESS\Actor_01\03-01-02-01-01-01-01.wav",   # calm
        r"D:\Projects\VoxAlign\Datasets\RAVDESS\Actor_01\03-01-03-01-01-01-01.wav",   # happy
        r"D:\Projects\VoxAlign\Datasets\RAVDESS\Actor_01\03-01-04-01-01-01-01.wav",   # sad
        r"D:\Projects\VoxAlign\Datasets\RAVDESS\Actor_01\03-01-05-01-01-01-01.wav",   # angry
        r"D:\Projects\VoxAlign\Datasets\RAVDESS\Actor_01\03-01-06-01-01-01-01.wav",   # fearful
        r"D:\Projects\VoxAlign\Datasets\RAVDESS\Actor_01\03-01-07-01-01-01-01.wav",   # disgust
        r"D:\Projects\VoxAlign\Datasets\RAVDESS\Actor_01\03-01-08-01-01-01-01.wav",   # surprised
        r"D:\Projects\VoxAlign\Datasets\SAVEE\ALL\DC_n22.wav",                         # example SAVEE
        # ... add more to reach 20
    ]

    # ---------- OPTION B: AUTO-DISCOVER ----------
    # Uncomment below to automatically collect 20 .wav files from a directory
    # dataset_roots = [
    #     r"D:\Projects\VoxAlign\Datasets\TESS",
    #     r"D:\Projects\VoxAlign\Datasets\RAVDESS",
    #     r"D:\Projects\VoxAlign\Datasets\SAVEE\ALL",
    # ]
    # all_wavs = []
    # for root in dataset_roots:
    #     all_wavs.extend(Path(root).rglob("*.wav"))
    # # Shuffle to get variety, then take first 20
    # import random
    # random.shuffle(all_wavs)
    # file_list = [str(p) for p in all_wavs[:20]]

    return file_list

# ================== MAIN EXECUTION ==================
if __name__ == "__main__":
    # Get the list of files to process
    test_files = collect_audio_files()
    # Limit to 20 if more are provided
    test_files = test_files[:20]
    print(f"📁 Processing {len(test_files)} files...\n")

    results = []
    for i, file_path in enumerate(test_files, 1):
        print(f"[{i}/{len(test_files)}] Processing: {file_path}")
        if not os.path.exists(file_path):
            print(f"⚠️  File not found, skipping.\n")
            continue
        result = analyze_emotion(file_path)
        if result:
            results.append(result)
            print(f"   ✅ {result['final_emotion']} (confidence: {result['confidence']:.1f}%)\n")

    # ----- Print Summary Table -----
    if results:
        print("\n" + "="*100)
        print("📊 MULTIMODAL EMOTION CLASSIFICATION SUMMARY")
        print("="*100)
        table_data = []
        for r in results:
            # Truncate transcription for display
            trans = r['transcription'][:50] + ('...' if len(r['transcription']) > 50 else '')
            table_data.append([
                r['file'],
                trans,
                r['audio_emotion'],
                r['text_emotion'],
                f"{r['final_emotion']} ({r['confidence']:.1f}%)"
            ])
        headers = ['File', 'Transcription', 'Audio Brain', 'Text Brain', 'Final Verdict']
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        print("="*100)
    else:
        print("No valid results to display.")