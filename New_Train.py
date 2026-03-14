import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import pandas as pd
import librosa
import numpy as np
import os
import random
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# ================= CONFIGURATION =================
MODEL_NAME = "facebook/wav2vec2-base-960h"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MASTER_CSV = r"D:\Projects\VoxAlign\Preprocessing\Preprocessig_outputs\FINAL_MULTIMODAL_DATASET.csv"
BATCH_SIZE = 4
LEARNING_RATE = 1e-5          # Base LR for classifier
ENCODER_LR = 1e-5              # LR for unfrozen encoder layers
UNFREEZE_LAST_LAYERS = 6        # Number of top encoder layers to unfreeze
EPOCHS = 30
NUM_CLASSES = 7
MAX_AUDIO_LEN = 16000 * 4       # 4 seconds at 16kHz
FOCAL_GAMMA = 2.0                # Gamma for focal loss
USE_FOCAL_LOSS = True

print(f"🔥 VoxAlign Enhanced Wav2Vec Engine on {DEVICE}")
print(f"Training on {MASTER_CSV}")
print(f"Unfreezing last {UNFREEZE_LAST_LAYERS} encoder layers, Focal Loss gamma={FOCAL_GAMMA}")

# ================= CLEAN CSV (REMOVE MISSING FILES) =================
def clean_csv(csv_path):
    """Remove rows where audio file does not exist."""
    df = pd.read_csv(csv_path)
    initial_len = len(df)
    existing = []
    for idx, row in df.iterrows():
        if Path(row['file_path']).exists():
            existing.append(row)
    df_clean = pd.DataFrame(existing)
    print(f"🧹 Cleaned CSV: {initial_len} -> {len(df_clean)} samples (removed {initial_len - len(df_clean)} missing files)")
    return df_clean

# ================= DATASET =================
class SimpleAudioDataset(Dataset):
    def __init__(self, csv_path, augment=False):
        # Clean the CSV on initialization
        self.data = clean_csv(csv_path)
        self.augment = augment
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

    def __len__(self):
        return len(self.data)

    def add_noise(self, audio, level=0.005):
        noise = np.random.randn(len(audio)) * level
        return audio + noise

    def time_stretch(self, audio):
        rate = random.uniform(0.9, 1.1)
        return librosa.effects.time_stretch(audio, rate=rate)

    def pitch_shift(self, audio, sr=16000):
        n_steps = random.randint(-2, 2)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = row['file_path']
        label = row['emotion_id']

        # Load audio (librosa returns mono by default)
        speech, _ = librosa.load(path, sr=16000)

        # Apply augmentations (only if training)
        if self.augment:
            # Noise
            if random.random() > 0.5:
                speech = self.add_noise(speech, level=random.uniform(0.001, 0.01))
            # Time stretch (with try/except to avoid failures on very short audio)
            if random.random() > 0.7:
                try:
                    speech = self.time_stretch(speech)
                except:
                    pass
            # Pitch shift
            if random.random() > 0.7:
                try:
                    speech = self.pitch_shift(speech, sr=16000)
                except:
                    pass

        # Fix length: trim or pad to MAX_AUDIO_LEN
        if len(speech) > MAX_AUDIO_LEN:
            if self.augment:
                start = random.randint(0, len(speech) - MAX_AUDIO_LEN)
            else:
                start = (len(speech) - MAX_AUDIO_LEN) // 2
            speech = speech[start:start + MAX_AUDIO_LEN]
        else:
            pad = MAX_AUDIO_LEN - len(speech)
            speech = np.pad(speech, (0, pad), 'constant')

        # Extract features using Wav2Vec2 feature extractor
        inputs = self.extractor(speech, sampling_rate=16000,
                                return_tensors="pt",
                                padding="max_length",
                                max_length=MAX_AUDIO_LEN,
                                truncation=True).input_values
        # inputs shape: (1, time) – remove the batch dimension to get (time,)
        inputs = inputs.squeeze(0)   # now shape (time,)

        return inputs, torch.tensor(label, dtype=torch.long)

# ================= MODEL WITH PARTIAL UNFREEZING =================
class VoxAlignWav2Vec(nn.Module):
    def __init__(self, num_classes=7, unfreeze_last=6):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(MODEL_NAME)
        # Freeze feature extractor (always, to save memory)
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False

        # Freeze/unfreeze encoder layers
        total_layers = len(self.wav2vec2.encoder.layers)
        for i, layer in enumerate(self.wav2vec2.encoder.layers):
            if i >= total_layers - unfreeze_last:
                for param in layer.parameters():
                    param.requires_grad = True
            else:
                for param in layer.parameters():
                    param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, time) – expected by Wav2Vec2Model
        outputs = self.wav2vec2(x).last_hidden_state
        pooled = torch.mean(outputs, dim=1)
        return self.classifier(pooled)

# ================= FOCAL LOSS =================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

# ================= TEMPERATURE SCALING =================
def calibrate_temperature(model, val_loader, device):
    """Find optimal temperature T on validation set."""
    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            logits_list.append(outputs.cpu())
            labels_list.append(labels.cpu())
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    T = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=50)

    def eval_func():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss
    optimizer.step(eval_func)
    return T.item()

# ================= TRAINING LOOP =================
def train():
    # Dataset will automatically clean the CSV
    dataset = SimpleAudioDataset(MASTER_CSV, augment=True)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # Ensure validation dataset does not use augmentation
    val_dataset.dataset.augment = False

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = VoxAlignWav2Vec(num_classes=NUM_CLASSES, unfreeze_last=UNFREEZE_LAST_LAYERS).to(DEVICE)

    # Separate parameter groups: classifier (higher LR), unfrozen encoder (lower LR)
    classifier_params = list(model.classifier.parameters())
    encoder_params = [p for n, p in model.named_parameters() if 'classifier' not in n and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': classifier_params, 'lr': LEARNING_RATE},
        {'params': encoder_params, 'lr': ENCODER_LR}
    ])

    if USE_FOCAL_LOSS:
        criterion = FocalLoss(gamma=FOCAL_GAMMA)
    else:
        criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

    best_val_acc = 0
    patience_counter = 0
    max_patience = 6

    print(f"🚀 Starting training on {len(train_dataset)} samples...")

    for epoch in range(EPOCHS):
        model.train()
        train_correct = 0
        train_total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for speech, labels in pbar:
            speech, labels = speech.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(speech)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            _, pred = outputs.max(1)
            train_total += labels.size(0)
            train_correct += pred.eq(labels).sum().item()
            pbar.set_postfix({'Acc': f'{100*train_correct/train_total:.2f}%'})

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for speech, labels in val_loader:
                speech, labels = speech.to(DEVICE), labels.to(DEVICE)
                outputs = model(speech)
                _, pred = outputs.max(1)
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds) * 100
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        print(f"📊 Epoch {epoch+1}: Val Acc = {val_acc:.2f}%, Val F1 = {val_f1:.4f}")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "VoxAlign_Wav2Vec_Enhanced_Best.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"⏹️ Early stopping after {epoch+1} epochs")
                break

    print(f"\n✅ Training complete! Best validation accuracy: {best_val_acc:.2f}%")

    # ========== Temperature Calibration ==========
    print("\n🌡️ Calibrating temperature on validation set...")
    best_model = VoxAlignWav2Vec(num_classes=NUM_CLASSES, unfreeze_last=UNFREEZE_LAST_LAYERS).to(DEVICE)
    best_model.load_state_dict(torch.load("VoxAlign_Wav2Vec_Enhanced_Best.pth", map_location=DEVICE))
    T = calibrate_temperature(best_model, val_loader, DEVICE)
    print(f"Optimal temperature T = {T:.4f}")
    torch.save({'model_state_dict': best_model.state_dict(), 'temperature': T},
               "VoxAlign_Wav2Vec_Enhanced_Calibrated.pth")
    print("✅ Calibrated model saved as VoxAlign_Wav2Vec_Enhanced_Calibrated.pth")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    train()