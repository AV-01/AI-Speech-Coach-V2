import streamlit as st
import ffmpeg
import tempfile
import os
import cv2
import numpy as np
from openai import OpenAI

def analyze_eye_contact(video_path):
    """Analyze eye contact and facial features in video"""
    # Load OpenCV classifiers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Analysis variables
    frames_with_face = 0
    frames_with_eyes = 0
    frames_with_eye_contact = 0
    total_processed_frames = 0
    
    # Process every 5th frame for efficiency
    frame_skip = 5
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Skip frames for efficiency
        if total_processed_frames % frame_skip != 0:
            total_processed_frames += 1
            continue
            
        # Update progress
        progress = min(total_processed_frames / total_frames, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {total_processed_frames}/{total_frames}")
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            frames_with_face += 1
            
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                
                # Filter eyes to only those in upper half of face
                valid_eyes = []
                for (ex, ey, ew, eh) in eyes:
                    if ey < h/2:  # Eye in upper half of face
                        valid_eyes.append((ex, ey, ew, eh))
                
                if len(valid_eyes) >= 2:  # Both eyes detected
                    frames_with_eyes += 1
                    
                    # Simple eye contact detection based on pupil position
                    eye_contact_detected = False
                    for (ex, ey, ew, eh) in valid_eyes:
                        eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
                        
                        # Threshold to find pupil
                        _, thresh = cv2.threshold(eye_gray, 42, 255, cv2.THRESH_BINARY_INV)
                        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if contours:
                            # Find largest contour (pupil)
                            largest_contour = max(contours, key=cv2.contourArea)
                            (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
                            
                            # Check if pupil is roughly centered (simple heuristic)
                            eye_center_x = ew / 2
                            eye_center_y = eh / 2
                            
                            # Allow some tolerance for "eye contact"
                            tolerance_x = ew * 0.3
                            tolerance_y = eh * 0.3
                            
                            if (abs(cx - eye_center_x) < tolerance_x and 
                                abs(cy - eye_center_y) < tolerance_y):
                                eye_contact_detected = True
                                break
                    
                    if eye_contact_detected:
                        frames_with_eye_contact += 1
        
        total_processed_frames += 1
        
        # Break if we've processed enough frames
        if total_processed_frames >= total_frames:
            break
    
    cap.release()
    progress_bar.empty()
    status_text.empty()
    
    # Calculate percentages
    face_presence = (frames_with_face / (total_processed_frames // frame_skip)) * 100 if total_processed_frames > 0 else 0
    eye_detection = (frames_with_eyes / frames_with_face) * 100 if frames_with_face > 0 else 0
    eye_contact_percentage = (frames_with_eye_contact / frames_with_eyes) * 100 if frames_with_eyes > 0 else 0
    
    return {
        "duration": duration,
        "total_frames_processed": total_processed_frames // frame_skip,
        "frames_with_face": frames_with_face,
        "frames_with_eyes": frames_with_eyes,
        "frames_with_eye_contact": frames_with_eye_contact,
        "face_presence_percentage": face_presence,
        "eye_detection_percentage": eye_detection,
        "eye_contact_percentage": eye_contact_percentage
    }

def analyze_transcription(transcription_json):
    segments = transcription_json.get("segments", [])
    
    total_words = 0
    total_duration = 0.0
    filler_words = {"um", "uh", "like", "you know", "so", "actually", "basically", "right", "okay", "well"}
    filler_count = 0
    
    segment_speeds = []  # list of tuples: (segment_index, wpm, start, end, text)
    
    for i, seg in enumerate(segments):
        # Get text and split into words
        text = seg.get("text", "").strip()
        if not text:
            continue
            
        words = text.split()
        num_words = len(words)
        duration = seg["end"] - seg["start"]
        
        total_words += num_words
        total_duration += duration if duration > 0 else 0.1  # small fallback to avoid zero division
        
        # Count filler words in this segment (case-insensitive)
        text_lower = text.lower()
        seg_filler = 0
        for filler in filler_words:
            # Count occurrences of each filler word
            if filler == "you know":
                # Special case for multi-word filler
                seg_filler += text_lower.count(filler)
            else:
                # Count as whole words to avoid false matches
                import re
                pattern = r'\b' + re.escape(filler) + r'\b'
                seg_filler += len(re.findall(pattern, text_lower))
        
        filler_count += seg_filler
        
        # Calculate WPM for segment
        wpm = (num_words / duration) * 60 if duration > 0 else 0
        segment_speeds.append((i, wpm, seg["start"], seg["end"], text))
    
    average_wpm = (total_words / total_duration) * 60 if total_duration > 0 else 0
    
    # Find fastest and slowest segment by WPM
    fastest_segment = max(segment_speeds, key=lambda x: x[1]) if segment_speeds else None
    slowest_segment = min(segment_speeds, key=lambda x: x[1]) if segment_speeds else None
    
    return {
        "total_words": total_words,
        "total_duration": total_duration,
        "average_wpm": average_wpm,
        "filler_count": filler_count,
        "filler_percentage": (filler_count / total_words * 100) if total_words > 0 else 0,
        "fastest_segment": fastest_segment,
        "slowest_segment": slowest_segment,
        "num_segments": len(segments)
    }


st.set_page_config(page_title="AI Speech Coach", layout='wide')
st.title("AI Speech Coach")
st.markdown("Upload a video to analyze both speech patterns and eye contact behavior")

uploaded = st.file_uploader("Upload MP4 or MOV video", type=['mp4', 'mov'])

# Add analysis options
col1, col2 = st.columns(2)
with col1:
    analyze_speech = st.checkbox("Analyze Speech", value=True)
with col2:
    analyze_vision = st.checkbox("Analyze Eye Contact & Facial Features", value=True)

run_button = st.button("Analyze")

if uploaded and run_button:
    if not analyze_speech and not analyze_vision:
        st.warning("Please select at least one analysis option.")
        st.stop()
    
    with st.spinner("Processing video..."):
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            temp_video_file.write(uploaded.read())
            temp_video_path = temp_video_file.name
        
        results = {}
        
        # Speech Analysis
        if analyze_speech:
            st.subheader("🎙️ Speech Analysis")
            with st.spinner("Extracting audio and transcribing..."):
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
                    results['speech'] = analyze_transcription(transcription_response.to_dict())
                except Exception as e:
                    st.error(f"Error during transcription: {e}")
                    transcription_text = None
                
                # Clean up audio file
                os.unlink(temp_audio_path)
        
        # Vision Analysis
        if analyze_vision:
            st.subheader("👁️ Eye Contact & Facial Analysis")
            with st.spinner("Analyzing facial features and eye contact..."):
                try:
                    results['vision'] = analyze_eye_contact(temp_video_path)
                except Exception as e:
                    st.error(f"Error during vision analysis: {e}")
                    results['vision'] = None
        
        # Clean up video file
        os.unlink(temp_video_path)
        
        # Display Results
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        # Create columns for results
        col1, col2 = st.columns(2)
        
        # Speech Results
        if analyze_speech and 'speech' in results:
            with col1:
                st.subheader("🎙️ Speech Metrics")
                speech_results = results['speech']
                
                # Key metrics
                st.metric("Average Words Per Minute", f"{speech_results['average_wpm']:.1f}")
                st.metric("Filler Words", speech_results['filler_count'])
                st.metric("Filler Word Percentage", f"{speech_results['filler_percentage']:.1f}%")
                
                # Performance insights
                if speech_results['average_wpm'] < 120:
                    st.info("💡 Consider speaking slightly faster for better engagement")
                elif speech_results['average_wpm'] > 180:
                    st.info("💡 Consider slowing down slightly for better clarity")
                else:
                    st.success("✅ Good speaking pace!")
                
                if speech_results['filler_percentage'] > 5:
                    st.warning("💡 Try to reduce filler words for more confident delivery")
                else:
                    st.success("✅ Good control of filler words!")
                
                # Segment details
                if speech_results['fastest_segment']:
                    i, wpm, start, end, text = speech_results['fastest_segment']
                    st.markdown("**Fastest segment:**")
                    st.markdown(f"⏱️ {start:.1f}s - {end:.1f}s ({wpm:.1f} WPM)")
                    st.markdown(f"📝 \"{text[:100]}...\"" if len(text) > 100 else f"📝 \"{text}\"")
                
                if speech_results['slowest_segment']:
                    i, wpm, start, end, text = speech_results['slowest_segment']
                    st.markdown("**Slowest segment:**")
                    st.markdown(f"⏱️ {start:.1f}s - {end:.1f}s ({wpm:.1f} WPM)")
                    st.markdown(f"📝 \"{text[:100]}...\"" if len(text) > 100 else f"📝 \"{text}\"")
        
        # Vision Results
        if analyze_vision and 'vision' in results:
            with col2:
                st.subheader("👁️ Visual Presence Metrics")
                vision_results = results['vision']
                
                # Key metrics
                st.metric("Face Visibility", f"{vision_results['face_presence_percentage']:.1f}%")
                st.metric("Eye Detection Rate", f"{vision_results['eye_detection_percentage']:.1f}%")
                st.metric("Eye Contact Estimate", f"{vision_results['eye_contact_percentage']:.1f}%")
                
                # Performance insights
                if vision_results['face_presence_percentage'] < 80:
                    st.warning("💡 Try to stay in frame more consistently")
                else:
                    st.success("✅ Good frame presence!")
                
                if vision_results['eye_detection_percentage'] < 70:
                    st.info("💡 Face the camera more directly for better eye detection")
                else:
                    st.success("✅ Good face positioning!")
                
                if vision_results['eye_contact_percentage'] < 60:
                    st.info("💡 Try to look at the camera more frequently")
                elif vision_results['eye_contact_percentage'] > 80:
                    st.success("✅ Excellent eye contact!")
                else:
                    st.success("✅ Good eye contact!")
                
                # Technical details
                st.markdown("**Technical Details:**")
                st.text(f"Video duration: {vision_results['duration']:.1f}s")
                st.text(f"Frames analyzed: {vision_results['total_frames_processed']}")
                st.text(f"Frames with face: {vision_results['frames_with_face']}")
                st.text(f"Frames with eyes: {vision_results['frames_with_eyes']}")
        
        # Combined insights
        if analyze_speech and analyze_vision and 'speech' in results and 'vision' in results:
            st.markdown("---")
            st.subheader("🎯 Overall Performance Summary")
            
            speech_score = min(100, max(0, 100 - results['speech']['filler_percentage'] * 2))
            vision_score = (results['vision']['face_presence_percentage'] + 
                          results['vision']['eye_contact_percentage']) / 2
            
            overall_score = (speech_score + vision_score) / 2
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Speech Score", f"{speech_score:.0f}/100")
            with col2:
                st.metric("Visual Score", f"{vision_score:.0f}/100")
            with col3:
                st.metric("Overall Score", f"{overall_score:.0f}/100")
            
            if overall_score >= 80:
                st.success("🌟 Excellent presentation skills!")
            elif overall_score >= 65:
                st.info("👍 Good presentation with room for improvement")
            else:
                st.warning("💪 Keep practicing - you're on the right track!")