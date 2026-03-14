import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-uncased"
NUM_CLASSES = 7
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 2e-5
DATASET_CSV = r"D:\Projects\VoxAlign\Preprocessing\Preprocessig_outputs\FINAL_MULTIMODAL_DATASET.csv"
MODEL_SAVE_PATH = "VoxAlign_TextBrain_BERT.pth"

print(f"🚀 Training BERT on {DEVICE}...")

# --- 2. DATA PREPARATION ---
class EmotionTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        # Handle placeholders
        if text in ["[SILENCE]", "[ERROR]", "nan", ""]:
            text = "neutral context"

        label = self.labels[item]

        # Use tokenizer directly (encode_plus is deprecated)
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(label, dtype=torch.long)
        }

print("📂 Loading Transcribed Dataset...")
df = pd.read_csv(DATASET_CSV)
df = df.dropna(subset=['emotion_id'])

# Split into train (80%) and validation (20%)
df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['emotion_id'])

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

train_dataset = EmotionTextDataset(df_train['transcription'].values, df_train['emotion_id'].values, tokenizer)
val_dataset = EmotionTextDataset(df_val['transcription'].values, df_val['emotion_id'].values, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# --- 3. MODEL SETUP ---
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES)
model = model.to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

# --- 4. TRAINING LOOP ---
best_accuracy = 0

for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
    
    # Training
    model.train()
    total_loss, correct_train, total_train = 0, 0, 0
    pbar = tqdm(train_loader, desc="Training")
    
    for batch in pbar:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        targets = batch['targets'].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs.logits, dim=1)
        correct_train += torch.sum(preds == targets).item()
        total_train += targets.size(0)
        pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
    train_acc = correct_train / total_train

    # Validation
    model.eval()
    correct_val, total_val = 0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            targets = batch['targets'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            correct_val += torch.sum(preds == targets).item()
            total_val += targets.size(0)

    val_acc = correct_val / total_val
    print(f"📊 Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    # Save the best model
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("💾 Best model saved!")

print(f"\n✅ Training Complete! Best Validation Accuracy: {best_accuracy*100:.2f}%")