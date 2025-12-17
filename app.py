import asyncio
import queue
import threading
import time

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

import assemblyai as aai

# ====== CONFIG ======
ASSEMBLYAI_API_KEY = "1169573ba261431c8104fc6deb3fcb42"
aai.settings.api_key = ASSEMBLYAI_API_KEY

st.set_page_config(page_title="Realtime Transcription", page_icon="🎙️")

st.title("🎙️ Realtime Transcription")
st.write("Speak into your mic and see live transcription from AssemblyAI.")

# Shared queue: audio from browser -> STT thread
audio_q: "queue.Queue[bytes]" = queue.Queue()

# ====== WebRTC audio processor (browser side) ======
class AudioProcessor(AudioProcessorBase):
    def recv(self, frame):
        # frame is an av.AudioFrame; get raw bytes (16‑bit PCM)
        pcm = frame.to_ndarray().tobytes()
        audio_q.put(pcm)
        return frame  # we are not changing audio

webrtc_ctx = webrtc_streamer(
    key="realtime-audio",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

# ====== STT worker using AssemblyAI realtime ======
transcript_placeholder = st.empty()

# make a place to accumulate text across callbacks
if "full_text" not in st.session_state:
    st.session_state["full_text"] = ""

def start_stt_loop():
    def on_data(t: aai.RealtimeTranscript):
        if not t.text:
            return
        if isinstance(t, aai.RealtimeFinalTranscript):
            st.session_state["full_text"] += t.text + " "
            transcript_placeholder.markdown(
                f"**You said:** {st.session_state['full_text']}"
            )

    def on_error(err: aai.RealtimeError):
        print("Realtime error:", err)

    rt = aai.RealtimeTranscriber(
        on_data=on_data,
        on_error=on_error,
        sample_rate=16000,
    )

    rt.connect()

    try:
        while webrtc_ctx.state.playing:
            try:
                pcm = audio_q.get(timeout=0.1)
            except queue.Empty:
                time.sleep(0.01)
                continue
            rt.send(pcm)
    finally:
        rt.close()

# ====== Start STT thread when WebRTC is running ======
if webrtc_ctx.state.playing:
    st.info("Listening… start speaking.")
    if "stt_thread" not in st.session_state:
        t = threading.Thread(target=start_stt_loop, daemon=True)
        st.session_state["stt_thread"] = t
        t.start()
else:
    st.warning("Click 'Start' in the WebRTC widget above to begin.")

