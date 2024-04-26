from flask import Flask, render_template, request
from pydub import AudioSegment
from pydub.utils import make_chunks
import os
import numpy as np
import librosa
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import nltk
from gtts import gTTS
import speech_recognition as sr
from langdetect import detect

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'uploads/'

try:
    os.makedirs("chunked")
except:
    pass

def preprocess_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    return audio

def extract_features(audio):
    audio_array = np.array(audio.get_array_of_samples()) / (2**15)
    mfccs = librosa.feature.mfcc(y=audio_array, sr=audio.frame_rate, n_mfcc=13)
    delta = librosa.feature.delta(mfccs)
    delta_delta = librosa.feature.delta(mfccs, order=2)
    features = np.vstack([mfccs, delta, delta_delta])
    return features.T

def cluster_audio(features):
    kmeans = KMeans(n_clusters=2, random_state=0)
    clusters = kmeans.fit_predict(features)
    return clusters

def generate_summary(transcription):
    # Split the transcription into sentences
    sentences = sent_tokenize(transcription)
    
    # Calculate the number of sentences to include in the summary (you can adjust this as needed)
    num_sentences = min(3, len(sentences))
    
    # Select the top sentences based on their importance (you can use various techniques here)
    important_sentences = sentences[:num_sentences]
    
    # Join the important sentences to form the summary
    summary = ' '.join(important_sentences)
    
    return summary


def transcribe_audio(audio_file, audio):
    # Initialize recognizer

    chunks = make_chunks(audio, 2000)
    text = ""
    for i, chunk in enumerate(chunks):
        chunkName= "./chunked/"+audio_file+"-{0}.wav".format(i)
        chunk.export(chunkName, format="wav")
        file = chunkName
        recognizer = sr.Recognizer()

        # Load audio file
        with sr.AudioFile(file) as source:
            audio_data = recognizer.record(source)

        # Recognize speech using Google Web Speech API
        try:
            text += recognizer.recognize_google(audio_data)
            print(text)
        except sr.UnknownValueError:
            print("Google Web Speech API could not understand the audio")
        except sr.RequestError as e:
            print("Could not request results from Google Web Speech API; {0}".format(e))
    return text

def generate_summary_audio(summary_text, output_path):
    summaries_folder = 'static/summaries'
    os.makedirs(summaries_folder, exist_ok=True)
    output_file = os.path.join(summaries_folder, "summary_audio.mp3")
    tts = gTTS(text=summary_text, lang='en')
    tts.save(output_file)
    return output_file

def identify_language(audio_text):
    try:
        language = detect(audio_text)
        return language
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
            summary = generate_summary(audio_text)  # Generate summary from the audio text
            generate_summary_audio(summary, "summary_audio.mp3")
            language = identify_language(audio_text)
            return render_template('index.html', transcription=audio_text, summary=summary, language=language)
    return render_template('index.html', summary=None, language=None)


if __name__ == '__main__':
    app.run(debug=True)
