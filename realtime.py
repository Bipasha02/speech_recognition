import pyaudio
import queue

# Real-time audio transcription
def live_audio_transcription(model, scorer):
    model = deepspeech.Model(model)
    model.enableExternalScorer(scorer)

    q = queue.Queue()

    def callback(in_data, frame_count, time_info, status):
        q.put(in_data)
        return (in_data, pyaudio.paContinue)

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, stream_callback=callback)
    stream.start_stream()

    print("Listening (press Ctrl+C to stop)...")
    try:
        while True:
            data = q.get()
            audio = np.frombuffer(data, dtype=np.int16)
            text = model.stt(audio)
            print("You said: ", text)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    model_file_path = 'deepspeech-0.9.3-models.pbmm'
    scorer_file_path = 'deepspeech-0.9.3-models.scorer'

    live_audio_transcription(model_file_path, scorer_file_path)
