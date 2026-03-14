# 🧠 VoxAlign: Multimodal Desktop AI Assistant

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![Gemini](https://img.shields.io/badge/AI-Google%20Gemini-teal)

VoxAlign is a state-of-the-art, multimodal desktop AI assistant. It combines real-time voice processing, emotional intelligence (via fine-tuned BERT), long-term vector memory, and Google's Gemini LLM to create a highly interactive, empathetic neural interface.

## ✨ Key Features
* **🎙️ Voice-First Interface:** Captures real-time audio and transcribes it instantly using **OpenAI Whisper**.
* **🧠 Emotional Intelligence:** Uses a custom, fine-tuned **BERT** model to analyze the user's emotional state from text/audio.
* **📚 RAG Memory Pipeline:** Integrates **ChromaDB** to retrieve contextual data (like customer records) for highly personalized AI responses.
* **🗣️ Neural Text-to-Speech:** Responds aloud using **Edge-TTS** for natural, fluid conversation.
* **🖥️ Custom 3D UI:** Built with `customtkinter` and `pygame`, featuring a futuristic, responsive visual core.

## 🛠️ Tech Stack
* **Machine Learning:** PyTorch, Transformers (Hugging Face), Scikit-Learn
* **AI Models:** Whisper (Audio), BERT (Emotion), Gemini (Logic)
* **Database:** ChromaDB (Vector Search)
* **UI & Audio:** CustomTkinter, OpenCV, SoundDevice, SoundFile

---

## 🚀 Installation & Setup

### 1. Prerequisites
* **Python 3.11** (Highly recommended)
* **FFmpeg** installed and added to your system PATH (Required for audio processing).
  * *Windows:* `winget install ffmpeg`

### 2. Clone the Repository
```bash
git clone [https://github.com/bhargavm-iiits/Vox-Align.git](https://github.com/bhargavm-iiits/Vox-Align.git)
cd Vox-Align
