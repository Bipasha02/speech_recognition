import numpy as np
import wave
import deepspeech
from scipy.io import wavfile
from pydub import AudioSegment

# Function to convert audio to the correct format
sample_file='sample_file'
def convert_audio(sample_file):
    audio = AudioSegment.from_file(sample_file)
    audio = audio.set_channels(1).set_frame_rate(16000)
    output_path = "converted_audio.wav"
    audio.export(output_path, format="wav")
    return output_path

# Function to read the WAV file
def read_wav_file(sample_file): 
    rate, data = wavfile.read(sample_file)
    return rate, data

# Function to transcribe audio
def transcribe_audio(sample_file, model, scorer):
    converted_audio_path = convert_audio(sample_file)
    model = deepspeech.Model(model)
    model.enableExternalScorer(scorer)

    rate, audio = read_wav_file(converted_audio_path)
    text = model.stt(audio)
    return text

# Main code
if __name__ == "__main__":
    model_file_path = 'deepspeech-0.9.3-models.pbmm'
    scorer_file_path = 'deepspeech-0.9.3-models.scorer'
    audio_file_path = 'sample_audio.wav'  

    print("Transcribing audio...")
    transcription = transcribe_audio(audio_file_path, model_file_path, scorer_file_path)
    print("Transcription:")
    print(transcription)
