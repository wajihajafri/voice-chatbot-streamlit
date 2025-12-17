import io
import time

import streamlit as st
import assemblyai as aai
from transformers import pipeline
from elevenlabs.client import ElevenLabs
import soundfile as sf


# ========= CONFIG =========
# In deployment, read these from environment variables instead of hardcoding.
ASSEMBLYAI_API_KEY = "1169573ba261431c8104fc6deb3fcb42"
ELEVENLABS_API_KEY = "sk_f4c9b0c67aa7e625a53c3ac4c9fda85a0f7b9608a171b2ab"
ELEVEN_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"
ELEVEN_MODEL_ID = "eleven_multilingual_v2"

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


# ========= TTS (ElevenLabs to WAV bytes) =========
@st.cache_resource
def get_eleven_client():
    return ElevenLabs(api_key=ELEVENLABS_API_KEY)

eleven_client = get_eleven_client()


def elevenlabs_tts_wav(text: str) -> bytes:
    # 1) Request audio from ElevenLabs (generator of bytes chunks)
    audio_stream = eleven_client.text_to_speech.convert(
        voice_id=ELEVEN_VOICE_ID,
        text=text,
        model_id=ELEVEN_MODEL_ID,
        output_format="pcm_22050",
    )

    # 2) Concatenate chunks
    audio_bytes = b"".join(chunk for chunk in audio_stream)

    # 3) Decode raw PCM to WAV bytes for Streamlit
    data, samplerate = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float32",
        samplerate=22050,
        channels=1,
        format="RAW",
        subtype="PCM_16",
    )

    buf = io.BytesIO()
    sf.write(buf, data, samplerate, format="WAV")
    return buf.getvalue()


# ========= STT (AssemblyAI, simple file mode) =========
def transcribe_with_assemblyai(wav_bytes: bytes) -> str:
    transcriber = aai.Transcriber()
    # Binary data is supported directly by transcribe()
    transcript = transcriber.transcribe(wav_bytes)
    return transcript.text or ""

# ========= STREAMLIT UI =========
st.title("Voice Chatbot (AssemblyAI + FLAN‑T5 + ElevenLabs)")

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
        reply_wav = elevenlabs_tts_wav(reply)
    st.audio(reply_wav, format="audio/wav")
elif process and audio_data is None:
    st.warning("Please record audio first.")
