import openai
import instaloader
from pytube import YouTube
import streamlit as st
import os
import subprocess
import streamlit.components.v1 as components
import streamlit_toggle as tog

def set_api_key():
    openai.api_key = st.secrets.openai_apikey

def tokens_to_brl(tokens):
    # Constants
    USD_TO_BRL = 5.0  # The conversion rate from USD to BRL
    API_COST_PER_TOKEN = 0.002 / 1000  # The OpenAI API cost per token
    return tokens * API_COST_PER_TOKEN * USD_TO_BRL

def get_video_url(link):
    try:
        if "instagram.com" in link:
            L = instaloader.Instaloader()
            post = instaloader.Post.from_shortcode(L.context, link.split('/')[-2])
            return post.video_url, "instagram"
        elif "youtube.com" in link or "youtu.be" in link:
            yt = YouTube(link)
            return yt.streams.filter(file_extension="mp4", mime_type="video/mp4", progressive=True).first().url, "youtube"
        else:
            return None, None
    except Exception as e:
        st.write("Wait a second and try again!")
        raise e
    
def copy_button(text):
    return components.html(
        open("copy_button/index.html").read().replace("{{ text }}", text.replace("\n", "\\n")),
        width=None,
        height=55,  # Set the height to 50 pixels
    )

# Add a new function to apply the user's prompt
def apply_prompt(prompt, text):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.1,
        messages=[
            {"role": "system", "content": "You are a helpful text formatter assistant that answers only using Markdown formatted text."},
            {"role": "user", "content": f"Apply the following prompt to the text:\n\n{prompt}\n\nText:\n{text}"},
        ]
    )
    tokens_used = completion["usage"]["total_tokens"]
    cost_in_brl = tokens_to_brl(tokens_used)
    return completion.choices[0].message.content, cost_in_brl

def process_transcription(transcription, temp=0.1) :
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature = temp,

    messages=[
            {"role": "system", "content": "You are a helpful text formatter assistant that answers only using Markdown formatted text."},
            {"role": "user", "content": f"Separate the text below into chapters. Use as many chapters as possible. Do not create a list of topics of chapter names. Include the whole text in its full length, but divided into chapters: {transcription}"},
        ]
    )
    tokens_used = completion["usage"]["total_tokens"]
    cost_in_brl = tokens_to_brl(tokens_used)
    return completion.choices[0].message.content, cost_in_brl

def create_title(transcription):
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature = 0.1,

    messages=[
            {"role": "system", "content": "You are a helpful text writing assistant that answers only using Markdown formatted text. You do not say or return anything else other than what was asked by the user"},
            {"role": "user", "content": f"Write a title for the text below. Return only the title, without quotes:\n\n{transcription}"},

        ]
    )
    tokens_used = completion["usage"]["total_tokens"]
    cost_in_brl = tokens_to_brl(tokens_used)
    return completion.choices[0].message.content.replace('"',''), cost_in_brl

def save_uploaded_file(uploaded_file):
    video_path = os.path.join(os.getcwd(), "user_uploaded_video.mp4")
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return video_path


def convert_video_to_audio(video_path):
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    with st.spinner('Converting video...'):
        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                audio_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    st.success("Video file uploaded and converted to audio.")
    return audio_path


def display_sidebar():
    with st.sidebar:
        st.subheader("Whisper Model Selection")
        model_options = [
            'tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small',
            'medium.en', 'medium', 'large-v1', 'large-v2', 'large'
        ]
        selected_model = st.selectbox('Select a model size', model_options)
        st.caption('The smaller the model, the faster the transcription.\nThe models with ".en" are smaller, but only work with the english language.')

        st.subheader("Language Selection")
        language_options = [
            'Auto detection', 'English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese'
        ]
        language_mapping = {
            'Auto detection': None,
            'English': 'en',
            'Spanish': 'es',
            'French': 'fr',
            'German': 'de',
            'Italian': 'it',
            'Portuguese': 'pt'
        }
        selected_language = st.selectbox('Select the language of the video', language_options)
        selected_language_code = language_mapping[selected_language]

        if selected_model.endswith('.en') and selected_language_code != 'en' and selected_language_code is not None:
            st.warning("The selected model only works with English. Please choose a different model or select English as the language.")

        st.subheader("Transcription Processing")
        st.caption('Keep this parameter turned off to see the transcription before processing it \n(Recommended)')
        process_transcription_toggle = tog.st_toggle_switch(
            label="Transcribe and process",
            key="Key2",
            default_value=False,
            label_after=True,
            inactive_color='#D3D3D3',
            active_color="#11567f",
            track_color="#29B5E8"
        )
    return selected_model, selected_language_code, process_transcription_toggle