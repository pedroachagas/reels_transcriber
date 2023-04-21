import streamlit as st
import instaloader
import numpy as np
import subprocess
import torch
import shutil
from IPython.display import display, Markdown
import whisper
import warnings
import openai

openai.api_key = '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'

def process_transcription(transcription):
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature = 0.1,

    messages=[
            {"role": "system", "content": "You are a helpful text formatter assistant that answers only using Markdown formatted text."},
            {"role": "user", "content": f"Separate the text below into chapters. Use as many chapters as possible. Do not create a list of topics of chapter names. Include the whole text in its full length, but divided into chapters: {transcription}"},
        ]
    )
    return completion.choices[0].message.content

def create_title(transcription):
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature = 0.1,

    messages=[
            {"role": "system", "content": "You are a helpful text writing assistant that answers only using Markdown formatted text. You do not say or return anything else other than what was asked by the user"},
            {"role": "user", "content": f"Write a title for the text below. Return only the title, without quotes:\n\n{transcription}"},

        ]
    )
    return completion.choices[0].message.content.replace('"','')

warnings.filterwarnings('ignore')

st.title("Instagram Reel Transcription")

L = instaloader.Instaloader()

link = st.text_input("Enter Reel link here:")

with st.sidebar:

    st.header("Model Selection")


    model_options = [
        'tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 
        'medium.en', 'medium', 'large-v1', 'large-v2', 'large'
    ]

    selected_model = st.selectbox('Select a model size', model_options)
    st.caption('The smaller the model, the faster the transcription.\nThe models with ".en" are smaller, but only work with the english language.')


if st.button("Download and Transcribe"):
    try:
        if link:

            whisper_model = whisper.load_model(selected_model)

            language = "Auto detection" 
            verbose = 'Live transcription' 
            task = 'transcribe' 
            temperature = 0.15 
            temperature_increment_on_fallback = 0.2 
            best_of = 5 
            beam_size = 8 
            patience = 1.0 
            length_penalty = -0.05 
            suppress_tokens = "-1" 
            initial_prompt = "" 
            condition_on_previous_text = True 
            fp16 = True
            compression_ratio_threshold = 2.4 #
            logprob_threshold = -1.0
            no_speech_threshold = 0.6 

            verbose_lut = {
            'Live transcription': False,
            'Progress bar': False,
            'None': None
            }

            args = dict(
            language = (None if language == "Auto detection" else language),
            verbose = verbose_lut[verbose],
            task = task,
            temperature = temperature,
            temperature_increment_on_fallback = temperature_increment_on_fallback,
            best_of = best_of,
            beam_size = beam_size,
            patience=patience,
            length_penalty=(length_penalty if length_penalty>=0.0 else None),
            suppress_tokens=suppress_tokens,
            initial_prompt=(None if not initial_prompt else initial_prompt),
            condition_on_previous_text=condition_on_previous_text,
            fp16=fp16,
            compression_ratio_threshold=compression_ratio_threshold,
            logprob_threshold=logprob_threshold,
            no_speech_threshold=no_speech_threshold
            )

            temperature = args.pop("temperature")
            temperature_increment_on_fallback = args.pop("temperature_increment_on_fallback")

            # Download video
            with st.spinner('Downloading video...'):
                post = instaloader.Post.from_shortcode(L.context, link.split('/')[-2])
                video_url = post.video_url


            with st.spinner('Transcribing video...'):
                transcription = whisper.transcribe(
                    whisper_model,
                    video_url,
                    temperature=temperature,
                    **args,
                    )['text']
            
            with st.spinner('Processing transcription...'):
                title = create_title(transcription)
                transcription_processed = process_transcription(transcription).replace('#', '##')
            
            st.success('Transcription completed!')

            with st.expander("See transcription"):
                st.markdown("# " + title)
                st.markdown(transcription_processed)
        else:
            st.write("Please fill out the field")

        
    except Exception as e:
        st.write("Something went wrong. Please try again later.")
        st.write(e)
