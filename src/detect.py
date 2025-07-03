import streamlit as st
import numpy as np
import librosa
import pyaudio
import joblib
import time

# Load pre-trained model
MODEL_PATH = "D:/Projects/baby_cry_detection/models/baby_cry_model.pkl"
model = joblib.load(MODEL_PATH)

# Audio configurations
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 2

# Feature extraction function
def extract_features(audio_data, sr):
    try:
        audio_data = librosa.effects.preemphasis(audio_data)
        non_silent_intervals = librosa.effects.split(audio_data, top_db=30)
        processed_audio = np.concatenate([audio_data[start:end] for start, end in non_silent_intervals])
        mfccs = librosa.feature.mfcc(y=processed_audio, sr=sr, n_mfcc=13)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled.reshape(1, -1)
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None

st.title("Baby Cry Detection")
st.write("This app detects baby crying in real-time using audio input or uploaded audio files.")

# Initialize session state
if "is_running" not in st.session_state:
    st.session_state.is_running = False

prediction_placeholder = st.empty()

# Real-time detection
if st.button("Start Realtime Detection") and not st.session_state.is_running:
    st.session_state.is_running = True
    st.info("Listening for baby crying... Press 'Stop Realtime Detection' to end.")
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        while st.session_state.is_running:
            frames = [stream.read(CHUNK, exception_on_overflow=False) for _ in range(int(RATE / CHUNK * RECORD_SECONDS))]
            audio_data = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
            features = extract_features(audio_data, RATE)
            if features is not None:
                prediction = model.predict(features)
                label = "Crying" if prediction[0] == 1 else "Not Crying"
                prediction_placeholder.write(f"Prediction: {label}")
            time.sleep(0.1)
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        st.session_state.is_running = False

if st.button("Stop Realtime Detection") and st.session_state.is_running:
    st.session_state.is_running = False
    st.info("Realtime detection stopped. Press 'Start Realtime Detection' to begin again.")

# File upload detection
st.write("---")
st.subheader("Upload Audio File")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    try:
        st.info("Processing the uploaded file...")
        # Load the audio file
        audio_data, sr = librosa.load(uploaded_file, sr=RATE)
        
        # Extract features
        features = extract_features(audio_data, sr)
        if features is not None:
            prediction = model.predict(features)
            label = "Crying" if prediction[0] == 1 else "Not Crying"
            st.success(f"Prediction: {label}")
    except Exception as e:
        st.error(f"Failed to process the uploaded file: {e}")

st.info("Press 'Start Realtime Detection' to listen in real-time or upload an audio file.")
