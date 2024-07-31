import streamlit as st
from pathlib import Path
import os
import WhisperGDrive as wp
from streamlit_mic_recorder import mic_recorder

@st.cache_resource
def load_whisper_model(model_type):
    if model_type == "Whisper English to English":
        return wp.load_whisper_model("openai/whisper-small.en",
                                     'https://drive.google.com/uc?id=14AVtj9MqoOeIcryPwG0Wv86hxE-MBS12',
                                     'whisper-tiny-en.tflite')
    elif model_type == "Whisper Multi-lingual":
        return wp.load_whisper_model("openai/whisper-small",
                                     'https://drive.google.com/uc?id=1V6vRfvCK4s7G0nM0Hpl_Djy_nNGcT3Hx',
                                     'whisper.tflite')

def save_voice(filename, data):
    directory = Path("voices")
    directory.mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filepath, "wb") as f:
        f.write(data)
    return filepath

def process_transcribing(file_name, file_contents, processor, interpreter, input_tensor, output_tensor):
    # Save the file to the "voices" folder
    file_path = save_voice(file_name, file_contents)
    filetype = str(file_name).split('.')[-1]
    if filetype == "mp4":
        output_directory = wp.create_mp4_audio_chunks(file_path)
    else:
        output_directory = wp.create_audio_chunks(file_path)
    st.success(f"File uploaded successfully!")

    # if st.button("Start Transcribing Process"):
    with st.spinner("Transcription in Process..."):
        if filetype == "mp4":
            transcript_text = wp.transcribe_mp4_audio_chunks(output_directory, processor, interpreter, input_tensor, output_tensor)
        else:
            transcript_text = wp.transcribe_audio_chunks(output_directory, processor, interpreter, input_tensor, output_tensor)
        transcript_text = ' '.join(transcript_text)
        st.write("Transcript")
        st.markdown(transcript_text)
    # os.remove(file_path)

def main():
    st.title("Whisper Speech To Text")

    # Add radio buttons for selecting the model type
    # model_type = st.radio("Select Whisper model:", ("Whisper en to en", "Whisper many to en"))
    model_type = st.selectbox("Select Whisper model:", ("Whisper English to English", "Whisper Multi-lingual"))

    # Load the selected model
    processor, interpreter, input_tensor, output_tensor = load_whisper_model(model_type)

    # Add radio buttons for selecting recording method
    recording_method = st.radio("Select recording method:", ("Upload file", "Record from microphone"))

    if recording_method == "Upload file":
        uploaded_file = st.file_uploader("Choose a voice file", type=["wav", "mp3", "flac", "mp4"])

        if uploaded_file is not None:
            # Read the contents of the file
            file_contents = uploaded_file.read()
            file_name = uploaded_file.name
            process_transcribing(file_name, file_contents, processor, interpreter, input_tensor, output_tensor)

    elif recording_method == "Record from microphone":
        # Hide file uploader
        st.info("Speak into the microphone...")

        audio = mic_recorder(
            start_prompt="Start recording",
            stop_prompt="Stop recording",
            just_once=False,
            use_container_width=False,
            callback=None,
            args=(),
            kwargs={},
            key=None
        )
        if audio is not None:
            # Save the recorded audio to file
            file_name = "recorded_audio.wav"
            file_contents = audio['bytes']
            process_transcribing(file_name, file_contents, processor, interpreter, input_tensor, output_tensor)

if __name__ == "__main__":
    main()
