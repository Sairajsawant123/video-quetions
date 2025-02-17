import streamlit as st
import openai
import whisper
import os
import pandas as pd

# Set up OpenAI API key (store it securely in environment variables or Streamlit secrets)
openai.api_key = st.secrets["OPENAI_API_KEY"]
import os
import subprocess
import streamlit as st

def download_ffmpeg():
    # Download FFmpeg binary
    url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-i686-static.tar.xz"
    ffmpeg_path = "/tmp/ffmpeg"
    
    if not os.path.exists(ffmpeg_path):
        st.write("Downloading FFmpeg...")
        subprocess.run(f"wget {url} -O ffmpeg.tar.xz", shell=True)
        subprocess.run("tar -xf ffmpeg.tar.xz -C /tmp", shell=True)
        subprocess.run("mv /tmp/ffmpeg*/ffmpeg /tmp/ffmpeg", shell=True)
    
    return ffmpeg_path

def set_ffmpeg_env():
    os.environ["PATH"] += os.pathsep + "/tmp"

# Download FFmpeg and set the environment
ffmpeg_path = download_ffmpeg()
set_ffmpeg_env()

# Function to transcribe the input video or audio file using Whisper
def transcribe_file(file_path):
    # Load the Whisper model (you can use a smaller model if needed)
    model = whisper.load_model("base")
    
    # Transcribe the file
    transcription = model.transcribe(file_path)
    
    return transcription['text']

# Function to send transcript to ChatGPT and generate questions and answers
def generate_questions_and_answers_from_transcript(transcript_text):
    prompt = f"Create fill-in-the-blank and match-the-following type questions from this transcript: \n\n{transcript_text}\n\nFor each question, provide the answer as well. Format the response as follows:\nQ: [Question]\nA: [Answer]\n\n"

    try:
        # Call OpenAI API 
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant and question creator for students."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except openai.OpenAIError as e:
        st.error(f"Error communicating with OpenAI: {e}")
        return None

# Function to parse questions and answers from GPT response
def parse_questions_and_answers(gpt_response):
    questions = []
    answers = []
    
    # Split the response into lines
    lines = gpt_response.split("\n")
    
    # Iterate through lines to extract questions and answers
    for i in range(len(lines)):
        if lines[i].startswith("Q:"):
            questions.append(lines[i][3:].strip())  # Extract question text
            if i + 1 < len(lines) and lines[i + 1].startswith("A:"):
                answers.append(lines[i + 1][3:].strip())  # Extract answer text
            else:
                answers.append("")  # If no answer is found, add an empty string
    
    return questions, answers

# Main function to handle file input and processing
def process_file(file_path):
    # Check if the file is valid
    if not os.path.exists(file_path):
        st.error(f"File {file_path} not found.")
        return
    
    # Transcribe the video/audio file
    st.write("Transcribing the file...")
    transcription_text = transcribe_file(file_path)
    
    # Display transcription
    st.subheader("Transcription:")
    st.write(transcription_text)
    
    # Generate questions and answers from transcript
    generate = st.radio("Generate questions?", ("Yes", "No"))
    if generate == "Yes":
        st.write("Generating questions and answers...")
        gpt_response = generate_questions_and_answers_from_transcript(transcription_text)
        if gpt_response:
            st.subheader("Generated Questions and Answers:")
            st.write(gpt_response)

            # Parse questions and answers
            questions, answers = parse_questions_and_answers(gpt_response)

            # Create a DataFrame
            df = pd.DataFrame({
                "Question": questions,
                "Answer": answers
            })

            # Display the DataFrame
            st.subheader("DataFrame with Questions and Answers:")
            st.dataframe(df)
        else:
            st.warning("No questions were generated.")

# Streamlit app layout
st.title("Video/Audio to Questions Generator")
st.write("Upload a video or audio file to transcribe it and generate questions and answers.")

# File uploader
uploaded_file = st.file_uploader("Choose a video or audio file", type=["mp3", "wav", "mp4", "m4a"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    file_path = os.path.join("/tmp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process the file
    process_file(file_path)
