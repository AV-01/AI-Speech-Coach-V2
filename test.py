import ffmpeg

def extract_audio(video_path, audio_path):
  ffmpeg.input(video_path).output(audio_path).run(overwrite_output=True)

video_file = "sample.mp4"
audio_file = "audio.wav"
extract_audio(video_file, audio_file)