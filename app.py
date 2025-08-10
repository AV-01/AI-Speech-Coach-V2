import streamlit as st
import ffmpeg
import tempfile
import os
from openai import OpenAI

def analyze_transcription(transcription_json):
    segments = transcription_json.get("segments", [])
    
    total_words = 0
    total_duration = 0.0
    filler_words = {"um", "uh", "like", "you know", "so", "actually", "basically", "right", "okay", "well"}
    filler_count = 0
    
    segment_speeds = []  # list of tuples: (segment_index, wpm, start, end, text)
    
    for i, seg in enumerate(segments):
        words = seg.get("words", [])
        num_words = len(words)
        duration = seg["end"] - seg["start"]
        total_words += num_words
        total_duration += duration if duration > 0 else 1  # avoid zero division
        
        # Count filler words in this segment
        seg_filler = sum(
            1 for w in words if w["word"].lower() in filler_words
        )
        filler_count += seg_filler
        
        # Calculate WPM for segment
        wpm = (num_words / duration) * 60 if duration > 0 else 0
        segment_speeds.append((i, wpm, seg["start"], seg["end"], seg["text"]))
    
    average_wpm = (total_words / total_duration) * 60 if total_duration > 0 else 0
    
    # Find fastest and slowest segment by WPM
    fastest_segment = max(segment_speeds, key=lambda x: x[1]) if segment_speeds else None
    slowest_segment = min(segment_speeds, key=lambda x: x[1]) if segment_speeds else None
    
    return {
        "average_wpm": average_wpm,
        "filler_count": filler_count,
        "fastest_segment": fastest_segment,
        "slowest_segment": slowest_segment
    }


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
                    response_format="verbose_json"
                )
            transcription_text = transcription_response
        except Exception as e:
            st.error(f"Error during transcription: {e}")
            transcription_text = None
        
        # Clean up temp files
        os.unlink(temp_video_path)
        os.unlink(temp_audio_path)
        
        if transcription_text:
            results = analyze_transcription(dict(transcription_response))

            st.markdown(f"**Average WPM:** {results['average_wpm']:.2f}")
            st.markdown(f"**Number of filler words:** {results['filler_count']}")

            if results['fastest_segment']:
                i, wpm, start, end, text = results['fastest_segment']
                st.markdown(f"**Fastest segment (WPM={wpm:.2f}):** {start:.2f}s - {end:.2f}s")
                st.text(text)

            if results['slowest_segment']:
                i, wpm, start, end, text = results['slowest_segment']
                st.markdown(f"**Slowest segment (WPM={wpm:.2f}):** {start:.2f}s - {end:.2f}s")
                st.text(text)

