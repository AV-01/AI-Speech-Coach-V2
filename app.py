import streamlit as st
import ffmpeg
import tempfile
import os
from openai import OpenAI

st.set_page_config(page_title="AI Speech Coach", layout='wide')
st.title("AI Speech Coach")

uploaded = st.file_uploader("Upload MP4 or MOV video", type=['mp4', 'mov'])
run_button = st.button("Analyze")

if uploaded and run_button:
    with st.spinner("Processing video..."):
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            temp_video_file.write(uploaded.read())
            temp_video_path = temp_video_file.name
        
        # Extract audio to temp mp3 file
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_audio_path = temp_audio_file.name
        temp_audio_file.close()  # Close so ffmpeg can write
        
        try:
            ffmpeg.input(temp_video_path).output(temp_audio_path, **{'q:a': 0, 'map': 'a?'}).run(overwrite_output=True)
        except Exception as e:
            st.error(f"Error extracting audio: {e}")
            os.unlink(temp_video_path)
            os.unlink(temp_audio_path)
            st.stop()
        
        # Transcribe with OpenAI Whisper
        try:
            client = OpenAI(api_key=st.secrets["openai_api_key"])
            with open(temp_audio_path, "rb") as audio_file:
                transcription_response = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1",
                    response_format="text"
                )
            transcription_text = transcription_response
        except Exception as e:
            st.error(f"Error during transcription: {e}")
            transcription_text = None
        
        # Clean up temp files
        os.unlink(temp_video_path)
        os.unlink(temp_audio_path)
        
        if transcription_text:
            st.subheader("Transcription")
            st.text_area("Transcribed Text", transcription_text, height=300)
