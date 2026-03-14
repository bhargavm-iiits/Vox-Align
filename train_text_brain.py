import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, BertTokenizer, BertForSequenceClassification
import librosa
import numpy as np

# --- Load Audio Model (Wav2Vec2) ---
class AudioModel(nn.Module):
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_model = AudioModel().to(device)
audio_model.load_state_dict(torch.load("VoxAlign_Wav2Vec_Best.pth", map_location=device))
audio_model.eval()

# --- Load Text Model (BERT) ---
text_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=7)
text_model.load_state_dict(torch.load("VoxAlign_TextBrain_BERT.pth", map_location=device))
text_model.to(device)
text_model.eval()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# --- Preprocessing for Audio ---
def preprocess_audio(audio_path):
    speech, _ = librosa.load(audio_path, sr=16000)
    max_len = 16000 * 4
    if len(speech) > max_len:
        speech = speech[:max_len]
    else:
        speech = np.pad(speech, (0, max_len - len(speech)), 'constant')
    # extractor needed – we assume you have it from earlier
    from transformers import Wav2Vec2FeatureExtractor
    extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
    inputs = extractor(speech, sampling_rate=16000, return_tensors="pt",
                       padding="max_length", max_length=max_len, truncation=True).input_values
    return inputs.to(device)

# --- Preprocessing for Text ---
def preprocess_text(transcription):
    if transcription in ["[SILENCE]", "[ERROR]", "nan", ""]:
        transcription = "neutral context"
    enc = tokenizer.encode_plus(transcription, max_length=128, padding='max_length',
                                 truncation=True, return_tensors='pt')
    return enc['input_ids'].to(device), enc['attention_mask'].to(device)

# --- Fusion Prediction ---
def predict_multimodal(audio_path, transcription):
    # Get audio logits
    audio_input = preprocess_audio(audio_path)
    with torch.no_grad():
        audio_logits = audio_model(audio_input)
        audio_probs = torch.softmax(audio_logits, dim=1)

    # Get text logits
    input_ids, attention_mask = preprocess_text(transcription)
    with torch.no_grad():
        text_outputs = text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_logits = text_outputs.logits
        text_probs = torch.softmax(text_logits, dim=1)

    # Average probabilities (late fusion)
    fused_probs = (audio_probs + text_probs) / 2.0
    pred = torch.argmax(fused_probs, dim=1).item()
    return pred, fused_probs.cpu().numpy()