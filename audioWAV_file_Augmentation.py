import os
import pandas as pd
from vosk import Model, KaldiRecognizer
import wave
import pyttsx3
import random
from transformers import pipeline

# Path to the directory where translated audio files will be saved
translated_audio_dir = "../data/augAudio_wav"
os.makedirs(translated_audio_dir, exist_ok=True)

# Load Vosk model for speech recognition
vosk_model_path = "/raid/home/adityalk/sid/tarun/clip/audio_model/vosk-api-0.3.50"  # Update this path
vosk_model = Model(vosk_model_path)

# Translation pipeline from transformers
translator_en_de = pipeline('translation', model='Helsinki-NLP/opus-mt-en-de')
translator_de_fr = pipeline('translation', model='Helsinki-NLP/opus-mt-de-fr')
translator_fr_en = pipeline('translation', model='Helsinki-NLP/opus-mt-fr-en')

def transcribe_audio(audio_path):
    wf = wave.open(audio_path, "rb")
    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        rec.AcceptWaveform(data)
    result = eval(rec.FinalResult())
    return result['text']

def translate_text(text, translator):
    result = translator(text, max_length=512)
    return result[0]['translation_text']

def text_to_speech(text, filename):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    selected_voice = random.choice(voices)
    engine.setProperty('voice', selected_voice.id)
    engine.save_to_file(text, filename)
    engine.runAndWait()

# Main function
def main():
    iemocap_path = "/raid/home/adityalk/sid/MMER/data/iemocap"
    for root, _, files in os.walk(iemocap_path):
        for file in files:
            if file.endswith(".wav"):
                audio_path = os.path.join(root, file)
                english_text = transcribe_audio(audio_path)
                german_text = translate_text(english_text, translator_en_de)
                french_text = translate_text(german_text, translator_de_fr)
                translated_english_text = translate_text(french_text, translator_fr_en)
                translated_audio_file = os.path.join(translated_audio_dir, file)
                text_to_speech(translated_english_text, translated_audio_file)
                print(f"Translated and saved audio: {translated_audio_file}")

if __name__ == "__main__":
    main()
