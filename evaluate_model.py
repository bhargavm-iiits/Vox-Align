import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import pandas as pd
import librosa
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
MODEL_NAME = "facebook/wav2vec2-base-960h"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "VoxAlign_Wav2Vec_Best.pth"
TEST_CSV = r"D:\Projects\VoxAlign\Preprocessing\Preprocessig_outputs\FINAL_AUDIO_DATASET_LOCAL.csv"  # your full dataset
BATCH_SIZE = 4
MAX_AUDIO_LEN = 16000 * 4

# Emotion mapping (0-6 to names)
emotion_names = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "angry",
    4: "fearful",
    5: "disgusted",
    6: "surprised"
}

# --- Dataset class (same as training, but without augmentation) ---
class SimpleAudioDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data.iloc[idx]['file_path']
        label = self.data.iloc[idx]['emotion_id']

        try:
            speech, _ = librosa.load(path, sr=16000)
            # Fix length
            if len(speech) > MAX_AUDIO_LEN:
                speech = speech[:MAX_AUDIO_LEN]
            else:
                speech = np.pad(speech, (0, MAX_AUDIO_LEN - len(speech)), 'constant')

            inputs = self.extractor(speech, sampling_rate=16000,
                                   return_tensors="pt",
                                   padding="max_length",
                                   max_length=MAX_AUDIO_LEN,
                                   truncation=True).input_values
            return inputs.squeeze(0), torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return dummy
            dummy = np.zeros(MAX_AUDIO_LEN)
            inputs = self.extractor(dummy, sampling_rate=16000,
                                   return_tensors="pt",
                                   padding="max_length",
                                   max_length=MAX_AUDIO_LEN,
                                   truncation=True).input_values
            return inputs.squeeze(0), torch.tensor(label, dtype=torch.long)

# --- Model definition (same as training) ---
class VoxAlignWav2Vec(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(MODEL_NAME)
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        outputs = self.wav2vec2(x)
        pooled = torch.mean(outputs.last_hidden_state, dim=1)
        return self.classifier(pooled)

# --- Load model ---
device = DEVICE
model = VoxAlignWav2Vec(num_classes=7).to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print(f"✅ Model loaded from {MODEL_PATH}")

# --- Load dataset (you may want to use a separate test CSV; here we use the whole set) ---
dataset = SimpleAudioDataset(TEST_CSV)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# --- Run inference ---
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(dataloader, desc="Evaluating"):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# --- Compute metrics ---
acc = accuracy_score(all_labels, all_preds) * 100
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"\n📊 Overall Accuracy: {acc:.2f}%")
print(f"📊 Weighted F1-Score: {f1:.4f}")

# --- Classification report ---
print("\n📋 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=[emotion_names[i] for i in range(7)]))

# --- Confusion matrix ---
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[emotion_names[i] for i in range(7)],
            yticklabels=[emotion_names[i] for i in range(7)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print("✅ Confusion matrix saved as confusion_matrix.png")