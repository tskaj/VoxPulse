from flask import Flask, render_template, request
from pydub import AudioSegment
from pydub.utils import make_chunks
import os
import numpy as np
import librosa
from sklearn.cluster import KMeans
from nltk.tokenize import sent_tokenize
from gtts import gTTS
import speech_recognition as sr
from langdetect import detect
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Create folders safely
os.makedirs("chunked", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

def preprocess_audio(file_path):
    """Convert audio to mono + 16kHz for consistency."""
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    return audio

def extract_features(audio):
    """Extract MFCC features for clustering."""
    audio_array = np.array(audio.get_array_of_samples()) / (2**15)
    mfccs = librosa.feature.mfcc(y=audio_array, sr=audio.frame_rate, n_mfcc=13)
    delta = librosa.feature.delta(mfccs)
    delta_delta = librosa.feature.delta(mfccs, order=2)
    features = np.vstack([mfccs, delta, delta_delta])
    return features.T

def cluster_audio(features):
    """Cluster audio frames into 2 groups (e.g., silence vs speech)."""
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)  # ✅ fixed warning
    clusters = kmeans.fit_predict(features)
    return clusters

def generate_summary(transcription):
    """Pick top 3 sentences as a simple extractive summary."""
    sentences = sent_tokenize(transcription)
    num_sentences = min(3, len(sentences))
    return ' '.join(sentences[:num_sentences])

def transcribe_audio(audio_file, audio):
    """Split audio into chunks & transcribe with Google API safely."""
    chunks = make_chunks(audio, 15000)  # 15 sec chunks is API sweet spot
    text = ""
    recognizer = sr.Recognizer()

    for i, chunk in enumerate(chunks):
        chunk_name = f"./chunked/{audio_file}-{i}.wav"
        chunk.export(chunk_name, format="wav")

        with sr.AudioFile(chunk_name) as source:
            audio_data = recognizer.record(source)

        try:
            result = recognizer.recognize_google(audio_data)
            text += result + ". "   # add punctuation for readability
            print(f"[OK] Processed chunk {i+1}/{len(chunks)}")
        except sr.UnknownValueError:
            print(f"[Skip] Could not understand chunk {i+1}/{len(chunks)}")
        except sr.RequestError as e:
            print(f"[Error] API request failed at chunk {i+1}: {e}")
            break   # stop on API failure

    return text.strip()

def generate_summary_audio(summary_text):
    """Convert summary into speech and save it."""
    summaries_folder = 'static/summaries'
    os.makedirs(summaries_folder, exist_ok=True)
    output_file = os.path.join(summaries_folder, "summary_audio.mp3")
    tts = gTTS(text=summary_text, lang='en')
    tts.save(output_file)
    return output_file

def identify_language(audio_text):
    """Detect language of the transcription."""
    try:
        return detect(audio_text)
    except:
        return "Language detection failed"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            audio = preprocess_audio(file_path)
            features = extract_features(audio)
            clusters = cluster_audio(features)

            audio_text = transcribe_audio(file.filename, audio)
            summary = generate_summary(audio_text)
            generate_summary_audio(summary)
            language = identify_language(audio_text)

            return render_template('index.html',
                                   transcription=audio_text,
                                   summary=summary,
                                   language=language)
    return render_template('index.html', summary=None, language=None)

if __name__ == '__main__':
    app.run(debug=True)
