import sys
import os

# --- CRITICAL FIX FOR --noconsole CRASH ---
if sys.stdout is None:
    class DummyStream:
        def write(self, *args, **kwargs): pass
        def flush(self, *args, **kwargs): pass
        def isatty(self): return False
    sys.stdout = DummyStream()
    sys.stderr = DummyStream()
# ------------------------------------------

import cv2 # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import sounddevice as sd # type: ignore
import soundfile as sf # type: ignore
import numpy as np # type: ignore
import whisper # type: ignore
import warnings
import chromadb # type: ignore
import pandas as pd
import re
import tempfile
import threading
import time
import queue
import csv
import math
from concurrent.futures import ThreadPoolExecutor
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, BertTokenizer, BertForSequenceClassification # type: ignore
import torch.nn.functional as F # type: ignore
from google import genai # type: ignore
import logging
import asyncio
import edge_tts # type: ignore
import pygame # type: ignore
import customtkinter as ctk # type: ignore
from PIL import Image, ImageTk # type: ignore
import random
from twilio.rest import Client # type: ignore
from dotenv import load_dotenv # type: ignore
# Load the hidden keys from the .env file
load_dotenv() 

# ==========================================
# ⚙️ 1. PATHS & TWILIO SETTINGS
# ==========================================
BASE_DIR = r"D:\Projects\VoxAlign\Preprocessing\Preprocessig_outputs"
CSV_PATH = os.path.join(BASE_DIR, "customers.csv")
WAV2VEC_WEIGHTS = os.path.join(BASE_DIR, "VoxAlign_Wav2Vec_Enhanced_Calibrated.pth")
BERT_WEIGHTS = os.path.join(BASE_DIR, "VoxAlign_TextBrain_BERT.pth")
DB_PATH = os.path.join(BASE_DIR, "voxalign_db")
EMOTION_CSV_PATH = os.path.join(BASE_DIR, "emotion_analytics.csv")

# 🔥 YOUR VIDEO FILE (muted playback)
VIDEO_PATH = r"C:\Users\Bhargav M\OneDrive\Pictures\Video_Ready_and_Available.mp4"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTIONS = {0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Angry', 4: 'Fear', 5: 'Disgust', 6: 'Surprise'}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')
pygame.mixer.init()

# ==========================================
# 🧠 2. AI CORE MODELS (Global Load)
# ==========================================
print("Loading AI Models... Please wait.")
client = genai.Client(api_key=GEMINI_API_KEY)
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

text_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=7).to(DEVICE)
text_model.load_state_dict(torch.load(BERT_WEIGHTS, map_location=DEVICE, weights_only=True))
text_model.eval()

chroma_client = chromadb.PersistentClient(path=DB_PATH)
memory_collection = chroma_client.get_or_create_collection(name="saas_production_db")

# ==========================================
# 🖥️ 3. DESKTOP GUI APP (CustomTkinter) – FULLSCREEN
# ==========================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class VoxAlignApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VoxAlign Intelligence Core")
        # Full‑screen (hides title bar)
        self.attributes('-fullscreen', True)

        self.current_phone = ""
        self.current_name = ""
        self.generated_otp = ""
        self.is_speaking = False
        self.running = True
        self.wave_animating = False
        self.wave_radius = 0

        # Video playback control
        self.video_cap = None
        self.video_thread_running = False

        # Lock for thread‑safe CSV writing
        self.emotion_log_lock = threading.Lock()

        # Main container fills the entire screen
        self.container = ctk.CTkFrame(self, fg_color="transparent")
        self.container.pack(fill="both", expand=True, padx=20, pady=20)

        # Bind Enter key globally
        self.bind('<Return>', self.handle_enter_key)

        self.show_login_screen()

    def handle_enter_key(self, event):
        """Trigger the active screen's submit action."""
        if hasattr(self, 'phone_entry') and self.phone_entry.winfo_exists():
            self.handle_send_otp()
        elif hasattr(self, 'otp_entry') and self.otp_entry.winfo_exists():
            self.handle_verify_otp()

    def update_orb_state(self, state):
        """Updates the status text to show what the AI is doing."""
        if not hasattr(self, 'status_label'): return
        colors = {
            "idle": "#3B8ED0",
            "listening": "#00E5FF",
            "thinking": "#D500F9",
            "speaking": "#00E676"
        }
        self.status_label.configure(text=state.upper() + "...", text_color=colors.get(state, "#FFFFFF"))

    # --- SCREEN 1: LOGIN ---
    def show_login_screen(self):
        for widget in self.container.winfo_children():
            widget.destroy()

        title = ctk.CTkLabel(self.container, text="SYSTEM ACCESS", font=("Consolas", 32, "bold"))
        title.pack(pady=(100, 30))

        desc = ctk.CTkLabel(self.container, text="Enter registered mobile identifier:", text_color="gray", font=("Consolas", 16))
        desc.pack(pady=10)

        self.phone_entry = ctk.CTkEntry(self.container, placeholder_text="e.g. 7795036940",
                                        width=400, height=60, font=("Consolas", 24), justify="center")
        self.phone_entry.pack(pady=20)
        self.phone_entry.focus_set()

        btn = ctk.CTkButton(self.container, text="SEND OTP", width=400, height=60,
                            font=("Consolas", 20, "bold"), command=self.handle_send_otp)
        btn.pack(pady=30)

        self.error_label = ctk.CTkLabel(self.container, text="", text_color="red", font=("Consolas", 16))
        self.error_label.pack()

    def handle_send_otp(self):
        phone = self.phone_entry.get().strip()
        try:
            df = pd.read_csv(CSV_PATH).dropna(subset=['phone', 'name'])
            df.columns = df.columns.str.strip().str.lower()
            df['phone'] = df['phone'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            match = df[df['phone'] == phone]

            if not match.empty:
                self.current_name = match.iloc[0]['name'].strip()
                self.current_phone = phone
                self.generated_otp = str(random.randint(1000, 9999))

                try:
                    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                    twilio_client.messages.create(
                        body=f"VoxAlign Security Code: {self.generated_otp}",
                        from_=TWILIO_PHONE_NUMBER,
                        to=f"+91{phone}"
                    )
                    print("✅ SMS Sent!")
                except Exception:
                    print(f"⚠️ Twilio SMS Failed. Fallback OTP: {self.generated_otp}")

                self.show_otp_screen()
            else:
                self.error_label.configure(text="Number not found in database.")
        except Exception as e:
            self.error_label.configure(text=f"Database Error: {e}")

    # --- SCREEN 2: OTP ---
    def show_otp_screen(self):
        for widget in self.container.winfo_children():
            widget.destroy()

        title = ctk.CTkLabel(self.container, text="AUTHORIZATION", font=("Consolas", 32, "bold"))
        title.pack(pady=(100, 30))

        self.otp_entry = ctk.CTkEntry(self.container, placeholder_text="••••",
                                      width=250, height=80, font=("Consolas", 40, "bold"), justify="center")
        self.otp_entry.pack(pady=20)
        self.otp_entry.focus_set()

        btn = ctk.CTkButton(self.container, text="VERIFY & CONNECT", width=400, height=60,
                            font=("Consolas", 20, "bold"), fg_color="#2E7D32", hover_color="#1B5E20",
                            command=self.handle_verify_otp)
        btn.pack(pady=30)

        self.otp_error = ctk.CTkLabel(self.container, text="", text_color="red", font=("Consolas", 16))
        self.otp_error.pack()

    def handle_verify_otp(self):
        if self.otp_entry.get().strip() == self.generated_otp:
            self.show_assistant_screen()
            threading.Thread(target=self.ai_voice_loop, daemon=True).start()
        else:
            self.otp_error.configure(text="Invalid authorization code.")

    # --- SCREEN 3: MAIN AI ASSISTANT (VIDEO BACKGROUND + WAVES) ---
    def show_assistant_screen(self):
        for widget in self.container.winfo_children():
            widget.destroy()

        header = ctk.CTkLabel(self.container, text="VOX CORE ONLINE", font=("Consolas", 28, "bold"), text_color="#00E5FF")
        header.pack(pady=(20, 10))

        # Canvas for video and waves – now uses relative sizing
        self.canvas = ctk.CTkCanvas(self.container, bg='#020205', highlightthickness=0)
        self.canvas.pack(pady=10, fill="both", expand=True)

        # Start video playback in a separate thread (muted)
        self.video_thread_running = True
        self.video_thread = threading.Thread(target=self.play_video, daemon=True)
        self.video_thread.start()

        self.status_label = ctk.CTkLabel(self.container, text="INITIALIZING...", font=("Consolas", 20, "bold"))
        self.status_label.pack(pady=10)

        # Chat Log – now uses relative sizing
        self.chat_log = ctk.CTkTextbox(self.container, font=("Consolas", 14), state="disabled", fg_color="#1E1E1E")
        self.chat_log.pack(pady=10, padx=20, fill="both", expand=True)

        self.log_message("SYSTEM", f"Authentication successful. Welcome, {self.current_name}.")

    def log_message(self, sender, text):
        self.chat_log.configure(state="normal")
        self.chat_log.insert("end", f"[{sender}] {text}\n\n")
        self.chat_log.see("end")
        self.chat_log.configure(state="disabled")

    # ==========================================
    # 🎥 VIDEO PLAYBACK (MUTED)
    # ==========================================
    def play_video(self):
        """Opens video file and continuously updates canvas with frames."""
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            logging.error(f"Could not open video file: {VIDEO_PATH}")
            self.canvas.after(0, lambda: self.canvas.create_text(
                self.canvas.winfo_width()//2, self.canvas.winfo_height()//2,
                text="VIDEO NOT AVAILABLE", fill="red", font=("Consolas", 20)))
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1.0 / fps if fps > 0 else 0.03

        while self.video_thread_running and self.running:
            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Convert frame from BGR to RGB, resize to canvas current size
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Get current canvas dimensions (may change if window resized)
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1:
                frame = cv2.resize(frame, (canvas_width, canvas_height))
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)

                # Update canvas on main thread
                self.canvas.after(0, self.update_video_frame, imgtk)

            time.sleep(delay)

        cap.release()

    def update_video_frame(self, imgtk):
        """Replace canvas image with new video frame."""
        self.canvas.delete("video")
        self.canvas.create_image(self.canvas.winfo_width()//2, self.canvas.winfo_height()//2,
                                 image=imgtk, tags="video")
        self.current_frame = imgtk

    # ==========================================
    # 🎙️ AUDIO & AI ENGINE LOGIC
    # ==========================================

    # ----- Emotion Logging to CSV (exact format as requested) -----
    def log_emotion_to_csv(self, user_text, emotion):
        """
        Append one row to emotion_analytics.csv with columns:
        timestamp (DD-MM-YYYY HH:MM), customer, message, emotion.
        """
        timestamp = time.strftime("%d-%m-%Y %H:%M")  # e.g. 11-03-2026 23:41
        row = [timestamp, self.current_name, user_text, emotion]
        file_exists = os.path.isfile(EMOTION_CSV_PATH)
        try:
            with self.emotion_log_lock:
                with open(EMOTION_CSV_PATH, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["timestamp", "customer", "message", "emotion"])
                    writer.writerow(row)
        except PermissionError:
            logging.warning(f"Cannot write to {EMOTION_CSV_PATH} – file may be open in another program.")
        except Exception as e:
            logging.warning(f"Failed to write emotion log: {e}")

    # ----- Wave Animation (rippling circles) -----
    def start_wave_animation(self):
        if self.wave_animating:
            return
        self.wave_animating = True
        self.wave_radius = 0
        self.animate_waves()

    def stop_wave_animation(self):
        self.wave_animating = False
        self.canvas.delete("wave")

    def animate_waves(self):
        """Draws rippling circles around the video (on top of video)."""
        if not self.wave_animating:
            return
        self.canvas.delete("wave")
        # Get center of canvas
        cx = self.canvas.winfo_width() // 2
        cy = self.canvas.winfo_height() // 2
        if cx < 10 or cy < 10:  # canvas not ready yet
            self.after(50, self.animate_waves)
            return
        # Create 3 ripples with different sizes
        for i in range(1, 4):
            r = (self.wave_radius + (i * 40)) % 200
            width = max(1, 4 - i)
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    outline="#38bdf8", width=width, tags="wave")
        self.wave_radius += 5
        self.after(50, self.animate_waves)

    # ----- TTS -----
    def speak(self, text):
        self.is_speaking = True
        self.after(0, self.update_orb_state, "speaking")
        self.after(0, self.log_message, "VOX", text)
        self.start_wave_animation()

        temp_audio = tempfile.mktemp(suffix=".mp3")
        try:
            communicate = edge_tts.Communicate(text, "en-US-ChristopherNeural", rate="-10%")
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
                try:
                    os.remove(temp_audio)
                except:
                    pass
            self.is_speaking = False
            self.stop_wave_animation()

    # ----- Recording (3-second silence after speech) -----
    def record_audio_clip(self, fs=16000, silence_timeout=3.0, max_duration=15.0):
        """
        Records until silence is detected for `silence_timeout` seconds after speech starts.
        Returns path to saved WAV file, or None if nothing recorded.
        """
        self.after(0, self.update_orb_state, "listening")
        self.start_wave_animation()

        q = queue.Queue()
        def callback(indata, frames, time, status):
            q.put(indata.copy())

        frames = []
        silent_chunks = 0
        started = False
        blocksize = int(fs * 0.05)  # 50 ms chunks
        max_chunks = int(max_duration / 0.05)

        with sd.InputStream(samplerate=fs, channels=1, dtype='int16', callback=callback, blocksize=blocksize):
            chunk_count = 0
            while self.running and not self.is_speaking and chunk_count < max_chunks:
                data = q.get()
                frames.append(data)
                chunk_count += 1

                rms = np.sqrt(np.mean(np.square(data.astype(np.float32))))
                threshold = 1800  # adjust for your mic

                if rms > threshold:
                    if not started:
                        started = True
                    silent_chunks = 0
                elif started:
                    silent_chunks += 1

                # Stop after silence_timeout seconds of silence after speech
                if started and silent_chunks >= int(silence_timeout / 0.05):
                    break

        self.stop_wave_animation()

        if not frames:
            return None

        audio = np.concatenate(frames)
        path = tempfile.mktemp(suffix=".wav")
        sf.write(path, audio, fs)
        return path

    # ----- Main AI Loop -----
    def ai_voice_loop(self):
        self.speak(f"Authorization accepted. Hello {self.current_name}, how may I assist you today?")

        while self.running:
            if self.is_speaking:
                time.sleep(0.1)
                continue

            audio_file = self.record_audio_clip()
            if not audio_file:
                continue

            self.after(0, self.update_orb_state, "thinking")
            result = whisper_model.transcribe(audio_file)
            text = result["text"].strip()
            if os.path.exists(audio_file):
                os.remove(audio_file)

            if len(text) > 4:
                self.after(0, self.log_message, self.current_name.upper(), text)

                if re.search(r"\b(exit|quit|bye|goodbye|thanks|thank you)\b", text.lower()):
                    self.speak("You are very welcome. Core systems powering down. Goodbye.")
                    self.running = False
                    self.after(1000, self.destroy)
                    break

                self.process_request(text)
            else:
                self.after(0, self.update_orb_state, "idle")

    def process_request(self, text):
        def get_emot():
            with torch.no_grad():
                inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                probs = F.softmax(text_model(**inputs).logits, dim=1)
                return EMOTIONS[torch.argmax(probs).item()].upper()

        emotion = executor.submit(get_emot).result()

        # Log the detected emotion in the exact required format
        self.log_emotion_to_csv(text, emotion)

        try:
            rag_results = memory_collection.query(query_texts=[text], n_results=1)
            policy = rag_results['documents'][0][0] if rag_results['documents'] else "Provide general support."
        except:
            policy = "Provide general support."

        prompt = f"""
        You are a highly intelligent, professional AI Assistant interacting via a voice interface.
        User Name: {self.current_name}
        Detected Emotion: {emotion}
        Company Policy Context: {policy}

        User says: "{text}"

        INSTRUCTIONS:
        - If the user asks about an order/company issue, use the Policy Context to help them.
        - If they ask general knowledge/tech questions, answer comprehensively like a standard AI.
        - Keep responses concise and natural for spoken audio.
        - Always end your response by asking if there is anything else you can help with.
        """

        try:
            response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            self.speak(response.text.strip())
        except Exception as e:
            logging.error(f"Gemini API Error: {e}")
            self.speak("I am experiencing a cognitive connection issue. Please repeat that.")

    def destroy(self):
        """Clean up video thread and release resources."""
        self.video_thread_running = False
        if hasattr(self, 'video_thread'):
            self.video_thread.join(timeout=1)
        super().destroy()

if __name__ == "__main__":
    app = VoxAlignApp()
    app.mainloop()