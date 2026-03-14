import pandas as pd
import whisper
import os
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Using your PERFECT dataset file!
INPUT_CSV = r"D:\Projects\VoxAlign\Preprocessing\Preprocessig_outputs\FINAL_AUDIO_DATASET_LOCAL.csv"
# Where the new text dataset will be saved
OUTPUT_CSV = r"D:\Projects\VoxAlign\Preprocessing\Preprocessig_outputs\FINAL_MULTIMODAL_DATASET.csv"

print(f"🚀 Initializing Whisper on: {DEVICE.upper()}")
model = whisper.load_model("base").to(DEVICE)

# --- 2. LOAD DATASET ---
print(f"📂 Loading dataset from {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)

# Create a 'transcription' column if it doesn't exist
if 'transcription' not in df.columns:
    df['transcription'] = None

# Find how many files still need transcribing (so you can pause/resume safely!)
pending_files = df[df['transcription'].isnull()]
print(f"🔍 Found {len(pending_files)} files to transcribe out of {len(df)} total.")

# --- 3. TRANSCRIPTION LOOP ---
SAVE_INTERVAL = 500  

try:
    for index, row in tqdm(pending_files.iterrows(), total=len(pending_files), desc="Transcribing"):
        audio_path = row['file_path']
        
        # Check if the file actually exists locally
        if not os.path.exists(audio_path):
            df.at[index, 'transcription'] = "[FILE MISSING]"
            continue
            
        try:
            # Let Whisper transcribe the audio
            result = model.transcribe(audio_path)
            text = result["text"].strip()
            df.at[index, 'transcription'] = text if text else "[SILENCE]"
        except Exception as e:
            df.at[index, 'transcription'] = "[ERROR]"

        # Auto-Save Checkpoint
        if (index + 1) % SAVE_INTERVAL == 0:
            df.to_csv(OUTPUT_CSV, index=False)
            
except KeyboardInterrupt:
    print("\n🛑 Transcription paused by user. Saving progress...")

# --- 4. FINAL SAVE ---
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Transcription run complete! File saved to:\n{OUTPUT_CSV}")