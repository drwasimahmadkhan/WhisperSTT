# # Import necessary libraries
# from transformers import WhisperProcessor
# import tensorflow as tf
# import whisper
# import numpy as np
# from timeit import default_timer as timer
#
# # Creating an instance of AutoProcessor from the pretrained model
# processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")
#
# # Define the path to the TFLite model
# tflite_model_path = 'assets/whisper-tiny-en.tflite'
#
# # Create an interpreter to run the TFLite model
# interpreter = tf.lite.Interpreter(tflite_model_path)
#
# # Allocate memory for the interpreter
# interpreter.allocate_tensors()
#
# # Get the input and output tensors
# input_tensor = interpreter.get_input_details()[0]['index']
# output_tensor = interpreter.get_output_details()[0]['index']
#
#
# inference_start = timer()
#
# # Calculate the mel spectrogram of the audio file
# print(f'Calculating mel spectrogram...')
# mel_from_file = whisper.audio.log_mel_spectrogram('assets/jfk.wav')
#
# # Pad or trim the input data to match the expected input size
# input_data = whisper.audio.pad_or_trim(mel_from_file, whisper.audio.N_FRAMES)
#
# # Add a batch dimension to the input data
# input_data = np.expand_dims(input_data, 0)
#
# # Run the TFLite model using the interpreter
# print("Invoking interpreter ...")
# interpreter.set_tensor(input_tensor, input_data)
# interpreter.invoke()
#
# # Get the output data from the interpreter
# output_data = interpreter.get_tensor(output_tensor)
#
# # Print the output data
# #print(output_data)
# transcription = processor.batch_decode(output_data, skip_special_tokens=True)[0]
# print(transcription)



# import librosa
#
# # Import necessary libraries
# from transformers import WhisperProcessor
# import tensorflow as tf
# import whisper
# import numpy as np
# from timeit import default_timer as timer
#
# # Creating an instance of AutoProcessor from the pretrained model
# processor = WhisperProcessor.from_pretrained("openai/whisper-small")
#
# # Define the path to the TFLite model
# tflite_model_path = 'assets/whisper-tiny-en.tflite'
#
# # Create an interpreter to run the TFLite model
# interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
# interpreter.allocate_tensors()
#
# # Get the input and output tensors
# input_tensor = interpreter.get_input_details()[0]['index']
# output_tensor = interpreter.get_output_details()[0]['index']
#
# # Process the audio file
# audio_file = 'assets/jfk.wav'
# mel_from_file = whisper.audio.log_mel_spectrogram(audio_file)
# input_data = whisper.audio.pad_or_trim(mel_from_file, whisper.audio.N_FRAMES)
# input_data = np.expand_dims(input_data, 0)
#
# # Run the TFLite model using the interpreter
# interpreter.set_tensor(input_tensor, input_data)
# interpreter.invoke()
#
# # Get the output data from the interpreter
# output_data = interpreter.get_tensor(output_tensor)
#
# # Decode and print the transcription
# transcription = processor.batch_decode(output_data, skip_special_tokens=True)[0]
# print(transcription)


#
#
# # Import necessary libraries
# import tensorflow as tf
# import numpy as np
# import soundfile as sf
# from transformers import WhisperProcessor
#
# # Define file paths
# tflite_model_path = 'assets/ml/whisper.tflite'
# audio_file_path = 'assets/ml/jfk.wav'
# vocab_file_path = 'assets/ml/filters_vocab_gen.bin'
#
# # Load and preprocess the audio file
# def load_audio(file_path):
#     audio, sample_rate = sf.read(file_path)
#     return audio, sample_rate
#
# def preprocess_audio(audio, sample_rate):
#     processor = WhisperProcessor.from_pretrained("openai/whisper-small")
#     inputs = processor(audio, sampling_rate=sample_rate, return_tensors="np")
#     return inputs.input_features
#
# # Load the audio file
# audio, sample_rate = load_audio(audio_file_path)
# input_data = preprocess_audio(audio, sample_rate)
#
# # Load the TFLite model and allocate tensors
# interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
# interpreter.allocate_tensors()
#
# # Get input and output tensor details
# input_tensor_index = interpreter.get_input_details()[0]['index']
# output_tensor_index = interpreter.get_output_details()[0]['index']
#
# # Prepare input data
# interpreter.set_tensor(input_tensor_index, input_data)
# interpreter.invoke()
#
# # Get the output data
# output_data = interpreter.get_tensor(output_tensor_index)
#
# # Decode the output data to get the transcription
# processor = WhisperProcessor.from_pretrained("openai/whisper-small")
# transcription = processor.batch_decode(output_data, skip_special_tokens=True)[0]
# print(transcription)


# ## Javed Code
# import tensorflow as tf
# import numpy as np
# import soundfile as sf
# from transformers import WhisperProcessor
# import os
# import tempfile
# import multiprocessing
# from pydub import AudioSegment
# from pydub.utils import make_chunks
# from scipy.signal import resample
#
# # Define file paths
# tflite_model_path = 'assets/ml/whisper.tflite'
# audio_file_path = 'mp3files/interview_video_27196_production.mp3'  # Update to .mp3 file
# output_dir = 'outputFiles'
# os.makedirs(output_dir, exist_ok=True)
#
#
# # Load and preprocess the audio file
# def load_audio(file_path):
#     audio = AudioSegment.from_file(file_path, format="mp3")  # Load MP3 file
#     return audio
#
#
# def resample_audio(audio, target_sample_rate=16000):
#     audio = audio.set_frame_rate(target_sample_rate)
#     return audio
#
#
# def preprocess_audio(audio, sample_rate):
#     processor = WhisperProcessor.from_pretrained("openai/whisper-small")
#     inputs = processor(audio, sampling_rate=sample_rate, return_tensors="np")
#     return inputs.input_features
#
#
# # Function to process a single audio chunk
# def process_chunk(chunk_path):
#     try:
#         audio, sample_rate = sf.read(chunk_path)
#         input_data = preprocess_audio(audio, sample_rate)
#
#         # Load the TFLite model and allocate tensors
#         interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
#         interpreter.allocate_tensors()
#
#         # Get input and output tensor details
#         input_tensor_index = interpreter.get_input_details()[0]['index']
#         output_tensor_index = interpreter.get_output_details()[0]['index']
#
#         # Prepare input data
#         interpreter.set_tensor(input_tensor_index, input_data)
#         interpreter.invoke()
#
#         # Get the output data
#         output_data = interpreter.get_tensor(output_tensor_index)
#
#         # Decode the output data to get the transcription
#         processor = WhisperProcessor.from_pretrained("openai/whisper-small")
#         transcription = processor.batch_decode(output_data, skip_special_tokens=True)[0]
#
#         # Remove the temporary chunk file
#         os.remove(chunk_path)
#
#         return transcription
#     except MemoryError as e:
#         print(f"MemoryError: {e}")
#         return ""
#
#
# def divide_audio_into_chunks(audio, chunk_length_ms):
#     chunks = make_chunks(audio, chunk_length_ms)
#     chunk_paths = []
#     for i, chunk in enumerate(chunks):
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
#             chunk.export(temp_file.name, format="wav")
#             chunk_paths.append(temp_file.name)
#     return chunk_paths
#
#
# if __name__ == '__main__':
#     # Load the audio file
#     audio = load_audio(audio_file_path)
#
#     # Resample the audio to 16000 Hz
#     resampled_audio = resample_audio(audio, 16000)
#
#     # Define chunk length (10 seconds to reduce memory load)
#     chunk_length_ms = 10 * 1000
#
#     # Divide audio into chunks
#     audio_chunks = divide_audio_into_chunks(resampled_audio, chunk_length_ms)
#
#     # Use multiprocessing to process chunks in parallel
#     with multiprocessing.Pool() as pool:
#         transcriptions = pool.map(process_chunk, audio_chunks)
#
#     # Combine all transcriptions
#     final_transcription = " ".join(transcriptions)
#     print(final_transcription)
#
#     # Write transcription to a file
#     output_file_name = os.path.splitext(os.path.basename(audio_file_path))[0] + ".txt"
#     output_file_path = os.path.join(output_dir, output_file_name)
#     with open(output_file_path, "w", encoding="utf-8") as f:
#         f.write(final_transcription)


####### Best code for MP3 files
'''
import os
from pydub import AudioSegment
from transformers import WhisperProcessor
import tensorflow as tf
import whisper
import numpy as np
from timeit import default_timer as timer
import re


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


def transcribe_audio_chunks(directory):
    # Creating an instance of AutoProcessor from the pretrained model
    processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")
    #processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    # Define the path to the TFLite model
    tflite_model_path = 'assets/whisper-tiny-en.tflite'
    #tflite_model_path = 'assets/ml/whisper.tflite'

    # Create an interpreter to run the TFLite model
    interpreter = tf.lite.Interpreter(tflite_model_path)

    # Allocate memory for the interpreter
    interpreter.allocate_tensors()

    # Get the input and output tensors
    input_tensor = interpreter.get_input_details()[0]['index']
    output_tensor = interpreter.get_output_details()[0]['index']

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


# Path to your input audio file
audio_file_path = 'mp3files/interview_video_26802_production.mp3'
#audio_file_path = 'hindiTimestamp/RES0034_H_chunks/chunk_1.wav'

# Create audio chunks and save them to the folder
output_directory = create_audio_chunks(audio_file_path)

# Perform transcription on the audio chunks
transcriptions = transcribe_audio_chunks(output_directory)

# Print the transcriptions
for i, transcription in enumerate(transcriptions):
    print(f"Chunk {i}: {transcription}")
'''


####### Best code for MP4 files

import os
from pydub import AudioSegment
from transformers import WhisperProcessor
import tensorflow as tf
import whisper
import numpy as np
from timeit import default_timer as timer
import re
import moviepy.editor as mp  # Added to handle mp4 files

def create_audio_chunks(file_path, chunk_length=26 * 1000, overlap=1 * 1000):
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

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def transcribe_audio_chunks(directory):
    # Creating an instance of AutoProcessor from the pretrained model
    processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")

    # Define the path to the TFLite model
    tflite_model_path = 'assets/whisper-tiny-en.tflite'

    # Create an interpreter to run the TFLite model
    interpreter = tf.lite.Interpreter(tflite_model_path)

    # Allocate memory for the interpreter
    interpreter.allocate_tensors()

    # Get the input and output tensors
    input_tensor = interpreter.get_input_details()[0]['index']
    output_tensor = interpreter.get_output_details()[0]['index']

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
