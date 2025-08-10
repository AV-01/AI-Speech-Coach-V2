import streamlit as st
import ffmpeg
import tempfile
import os
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
        
        if transcription_text:
            results = analyze_transcription(transcription_response.to_dict())
            # st.write(transcription_response.to_dict())
            st.write("## Speech Analysis Results")
            st.markdown(f"Average WPM: {results['average_wpm']:.2f}")
            st.markdown(f"Number of filler words: {results['filler_count']}")

            if results['fastest_segment']:
                i, wpm, start, end, text = results['fastest_segment']
                st.markdown(f"Fastest segment (WPM={wpm:.2f}): {start:.2f}s - {end:.2f}s")
                st.text(text)

            if results['slowest_segment']:
                i, wpm, start, end, text = results['slowest_segment']
                st.markdown(f"Slowest segment (WPM={wpm:.2f}): {start:.2f}s - {end:.2f}s")
                st.text(text)

        try:
            results['vision'] = analyze_eye_contact(temp_video_path)
        except Exception as e:
            st.error(f"Error during vision analysis: {e}")
            results['vision'] = None


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

        # Clean up temp files
        os.unlink(temp_video_path)
        os.unlink(temp_audio_path)