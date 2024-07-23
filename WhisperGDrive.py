# import os
# from pydub import AudioSegment
# from transformers import WhisperProcessor
# import tensorflow as tf
# import whisper
# import numpy as np
# from timeit import default_timer as timer
# import re
# import moviepy.editor as mp  # Added to handle mp4 files
# import gdown



import os
from pydub import AudioSegment
from transformers import WhisperProcessor
import tensorflow as tf
import whisper
import numpy as np
from timeit import default_timer as timer
import re
import gdown
import moviepy.editor as mp  # Added to handle mp4 files
def load_whisper_model(processor_name, url, tflite_model_path):
    # Creating an instance of AutoProcessor from the pretrained model
    processor = WhisperProcessor.from_pretrained(processor_name)

    # # Google Drive link in the correct format
    # url = 'https://drive.google.com/uc?id=14AVtj9MqoOeIcryPwG0Wv86hxE-MBS12'
    #
    # # Output file name for the downloaded model
    # tflite_model_path = 'whisper-tiny-en.tflite'

    # Download the TFLite model
    gdown.download(url, tflite_model_path, quiet=False)

    # Load the TFLite model using the TensorFlow Lite interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

    # Allocate memory for the interpreter
    interpreter.allocate_tensors()

    # Get the input and output tensors
    input_tensor = interpreter.get_input_details()[0]['index']
    output_tensor = interpreter.get_output_details()[0]['index']

    return processor, interpreter, input_tensor, output_tensor
def create_audio_chunks(file_path, chunk_length=26 * 1000, overlap=1 * 1000):
    # Load the audio file
    audio = AudioSegment.from_file(file_path)

    # Ensure the audio is at 16000 Hz
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)

    # Create a directory named after the audio file
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = f"{file_name}_chunks"
    os.makedirs(output_dir, exist_ok=True)

    # Calculate chunks
    start = 0
    end = chunk_length
    chunk_number = 0

    while start < len(audio):
        chunk = audio[start:end]
        chunk.export(f"{output_dir}/{file_name}_chunk_{chunk_number}.mp3", format="mp3")
        chunk_number += 1
        start = end - overlap
        end = start + chunk_length

    return output_dir


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def transcribe_audio_chunks(directory, processor, interpreter, input_tensor, output_tensor):

    transcriptions = []

    chunk_files = sorted(os.listdir(directory), key=natural_sort_key)
    for chunk_file in chunk_files:
        if chunk_file.endswith(".mp3"):
            file_path = os.path.join(directory, chunk_file)
            print(f"Processing file: {file_path}")

            # Calculate the mel spectrogram of the audio file
            mel_from_file = whisper.audio.log_mel_spectrogram(file_path)

            # Pad or trim the input data to match the expected input size
            input_data = whisper.audio.pad_or_trim(mel_from_file, whisper.audio.N_FRAMES)

            # Add a batch dimension to the input data
            input_data = np.expand_dims(input_data, 0)

            # Run the TFLite model using the interpreter
            interpreter.set_tensor(input_tensor, input_data)
            interpreter.invoke()

            # Get the output data from the interpreter
            output_data = interpreter.get_tensor(output_tensor)

            # Decode the transcription
            transcription = processor.batch_decode(output_data, skip_special_tokens=True)[0]
            transcriptions.append(transcription)

    return transcriptions

def transcribe_mp4_audio_chunks(directory, processor, interpreter, input_tensor, output_tensor):

    transcriptions = []

    chunk_files = sorted(os.listdir(directory), key=natural_sort_key)
    for chunk_file in chunk_files:
        if chunk_file.endswith(".mp3"):
            file_path = os.path.join(directory, chunk_file)
            print(f"Processing file: {file_path}")

            # Calculate the mel spectrogram of the audio file
            mel_from_file = whisper.audio.log_mel_spectrogram(file_path)

            # Pad or trim the input data to match the expected input size
            input_data = whisper.audio.pad_or_trim(mel_from_file, whisper.audio.N_FRAMES)

            # Add a batch dimension to the input data
            input_data = np.expand_dims(input_data, 0)

            # Run the TFLite model using the interpreter
            interpreter.set_tensor(input_tensor, input_data)
            interpreter.invoke()

            # Get the output data from the interpreter
            output_data = interpreter.get_tensor(output_tensor)

            # Decode the transcription
            transcription = processor.batch_decode(output_data, skip_special_tokens=True)[0]
            transcriptions.append(transcription)

    return transcriptions


def create_mp4_audio_chunks(file_path, chunk_length=26 * 1000, overlap=1 * 1000):
    # Load the video file
    video = mp.VideoFileClip(file_path)

    # Extract the audio from the video
    audio = video.audio
    frame_rate = audio.fps
    if frame_rate != 16000:
        audio = audio.set_fps(16000)
        frame_rate = 16000

    # Create a directory named after the audio file
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = f"{file_name}_chunks"
    os.makedirs(output_dir, exist_ok=True)

    # Calculate chunks
    start = 0
    end = chunk_length / 1000  # convert milliseconds to seconds
    chunk_number = 0

    while start < video.duration:
        chunk = audio.subclip(start, min(end, video.duration))
        chunk_file_path = f"{output_dir}/{file_name}_chunk_{chunk_number}.mp3"
        chunk.write_audiofile(chunk_file_path, codec="mp3", fps=frame_rate)
        chunk_number += 1
        start = end - (overlap / 1000)
        end = start + (chunk_length / 1000)

    return output_dir






if __name__ == '__main__':
    # Path to your input video file
    video_file_path = 'mp3files/safm.mp4'

    # Create audio chunks and save them to the folder
    output_directory = create_audio_chunks(video_file_path)

    # Perform transcription on the audio chunks
    transcriptions = transcribe_audio_chunks(output_directory)

    # Print the transcriptions
    for i, transcription in enumerate(transcriptions):
        print(f"{transcription}")
        #print(f"Chunk {i}: {transcription}")
