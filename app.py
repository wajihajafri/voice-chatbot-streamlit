import io
import time

import streamlit as st
import assemblyai as aai
from transformers import pipeline
from openai import OpenAI
import soundfile as sf

# ========= CONFIG =========
# In deployment, read these from Streamlit secrets.
ASSEMBLYAI_API_KEY = st.secrets["1169573ba261431c8104fc6deb3fcb42"]
OPENAI_API_KEY = st.secrets["sk-proj-D22xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"]

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
    # OpenAI Audio API, GPT‑4o mini TTS model
    resp = openai_client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
        format="wav",
    )
    # resp is a streaming object; read() returns WAV bytes
    return resp.read()


# ========= STT (AssemblyAI, simple file mode) =========
def transcribe_with_assemblyai(wav_bytes: bytes) -> str:
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(wav_bytes)
    return transcript.text or ""


# ========= STREAMLIT UI =========
st.title("Voice Chatbot (AssemblyAI + FLAN‑T5 + OpenAI TTS)")

st.markdown(
    "1. Click below to record your voice.\n"
    "2. Then click **Process** to get transcription, text reply, and spoken reply."
)

audio_data = st.audio_input("Speak your question here", sample_rate=16000)
process = st.button("Process")

if process and audio_data is not None:
    wav_bytes = audio_data.getvalue()

    with st.spinner("Transcribing with AssemblyAI..."):
        user_text = transcribe_with_assemblyai(wav_bytes)
    st.write(f"**You said:** {user_text}")

    with st.spinner("Generating reply (FLAN‑T5)..."):
        reply = chat_response(user_text)
    st.write(f"**Bot:** {reply}")

    with st.spinner("Speaking reply..."):
        reply_wav = tts_wav(reply)
    st.audio(reply_wav, format="audio/wav")

elif process and audio_data is None:
    st.warning("Please record audio first.")

