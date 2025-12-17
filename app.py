import io
import time

import streamlit as st
import assemblyai as aai
from transformers import pipeline
from openai import OpenAI
import soundfile as sf

# ========= PAGE CONFIG & HEADER =========
st.set_page_config(
    page_title="AI Voice Assistant",
    page_icon="🎧",
    layout="centered",
)

st.markdown(
    """
    <style>
    .app-title {
        font-size: 40px;
        font-weight: 800;
        letter-spacing: 0.02em;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        color: #6c757d;
        font-size: 15px;
        margin-bottom: 1.5rem;
    }
    .app-subtitle span {
        font-weight: 600;
        color: #4f46e5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-title">AI Voice Assistant</div>
    <div class="app-subtitle">
        Speak naturally and get instant answers –
        <span>AssemblyAI · FLAN‑T5 · OpenAI TTS</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ========= CONFIG =========
ASSEMBLYAI_API_KEY = st.secrets["ASSEMBLYAI_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

aai.settings.api_key = ASSEMBLYAI_API_KEY

# ========= LLM (FLAN‑T5) =========
@st.cache_resource
def load_chatbot():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
    )

chatbot = load_chatbot()


def chat_response(user_text: str) -> str:
    prompt = (
        "You are a knowledgeable and helpful assistant. Answer the user's question as accurately "
        "and informatively as possible. If you don't know the exact answer, provide the best "
        "possible response.\n"
        f"Question: {user_text}\nAnswer:"
    )
    out = chatbot(prompt, max_new_tokens=150, no_repeat_ngram_size=2)
    return out[0]["generated_text"].strip()

# ========= TTS (OpenAI to WAV bytes) =========
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)

openai_client = get_openai_client()


def tts_wav(text: str) -> bytes:
    resp = openai_client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
        response_format="wav",
    )
    return resp.read()

# ========= STT (AssemblyAI, simple file mode) =========
def transcribe_with_assemblyai(wav_bytes: bytes) -> str:
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(wav_bytes)
    return transcript.text or ""

# ========= STREAMLIT UI =========
st.markdown("### 🎤 Ask your question")
st.write("1. Record your question below.")
st.write("2. Click **Process** to get transcription, text reply, and spoken reply.")

audio_data = st.audio_input("Speak here", sample_rate=16000)
process = st.button("Process", type="primary")

if process and audio_data is not None:
    wav_bytes = audio_data.getvalue()

    with st.spinner("Transcribing with AssemblyAI..."):
        user_text = transcribe_with_assemblyai(wav_bytes)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### You said")
        st.write(user_text)

    with st.spinner("Generating reply (FLAN‑T5)..."):
        reply = chat_response(user_text)

    with col2:
        st.markdown("#### Bot reply")
        st.write(reply)

    with st.spinner("Speaking reply..."):
        reply_wav = tts_wav(reply)

    st.markdown("#### 🔊 Listen")
    st.audio(reply_wav, format="audio/wav")

elif process and audio_data is None:
    if process:
        st.warning("Please record audio first.")
