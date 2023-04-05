import streamlit as st
import instaloader
import numpy as np
import subprocess
import torch
import shutil
from IPython.display import display, Markdown
import whisper
import warnings

warnings.filterwarnings('ignore')

st.title("Instagram Reel Transcription")

link = st.text_input("Enter Reel link here:")

model_options = [
    'tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 
    'medium.en', 'medium', 'large-v1', 'large-v2', 'large'
]

selected_model = st.selectbox('Select a model configuration', model_options)

if st.button("Download and Transcribe"):
    try:
        if link:
            whisper_model = whisper.load_model(selected_model)

            if selected_model in whisper.available_models():
                st.write(f"**{selected_model} model is selected.**")
            else:
                st.write(f"**{selected_model} model is no longer available.**")
                
            language = "Auto detection" 
            verbose = 'Live transcription' 
            output_format = 'all' 
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
            if temperature_increment_on_fallback is not None:
                temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
            else:
                temperature = [temperature]


            if selected_model.endswith(".en") and args["language"] not in {"en", "English"}:
                warnings.warn(f"{selected_model} is an English-only model but receipted '{args['language']}'; using English instead.")
                args["language"] = "en"

            # Download video
            L = instaloader.Instaloader()
            post = instaloader.Post.from_shortcode(L.context, link.split('/')[-2])
            video_url = post.video_url
            transcription = whisper.transcribe(
                whisper_model,
                video_url,
                temperature=temperature,
                **args,
            )['text']
            st.write(f"Transcription: {transcription}")
        else:
            st.write("Please fill out the field")
            
    except Exception as e:
        st.write("Something went wrong. Please try again later.")
        st.write(e)
