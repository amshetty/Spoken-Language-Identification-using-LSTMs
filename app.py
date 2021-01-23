import streamlit as st
from tensorflow.keras.models import load_model
from clean import downsample_mono, envelope
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
from pydub import AudioSegment
import numpy as np
import os
from tqdm import tqdm
import joblib

page_bg_img = '''
<style>
body {
background-image: url("https://images.pexels.com/photos/131634/pexels-photo-131634.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=1020");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)
st.title('Spoken Language Identification (English/Hindi/Mandarin)')
st.sidebar.subheader('Please upload an image file')
st.sidebar.text('Select Input Audio file format')
check_wav = st.sidebar.checkbox('WAV')
check_mp3 = st.sidebar.checkbox('MP3')
audio = None
if check_wav:
    audio = st.sidebar.file_uploader('Upload .wav audio file', type = ['wav'] )

if check_mp3:
    temp = st.sidebar.file_uploader('Upload audio file', type = ['mp3'] )
    if temp:
        joblib.dump(temp, 'aud.mp3')
        sound = AudioSegment.from_mp3('./aud.mp3')
        sound.export('./audio.wav', format = 'wav')
        audio = './audio.wav'
model = load_model('lstm.h5', custom_objects={'STFT':STFT,
                'Magnitude':Magnitude,
                'ApplyFilterbank':ApplyFilterbank,
                'MagnitudeToDecibel':MagnitudeToDecibel})

classes = ['English', 'Hindi', 'Mandarin']

if audio:
    st.info('Analyzing audio file')
    rate, wav = downsample_mono(audio, 16000)
    mask, env = envelope(wav, rate, threshold = 10)
    clean_wav = wav[mask]
    step = int(16000*10)
    batch = []
    for i in range(0, clean_wav.shape[0], step):
        sample = clean_wav[i:i+step]
        sample = sample.reshape(-1,1)
        if sample.shape[0] < step:
            tmp = np.zeros(shape=(step, 1), dtype=np.float32)
            tmp[:sample.shape[0],:] = sample.flatten().reshape(-1,1)
            sample = tmp
        batch.append(sample)
    X_batch = np.array(batch, dtype=np.float32)
    y_pred = model.predict(X_batch)
    y_mean = np.mean(y_pred, axis = 0)
    y_pred = np.argmax(y_mean)
    st.success(f'Language Detected -->{classes[y_pred]}')
    os.remove('./aud.mp3')
    os.remove('./audio.wav')
