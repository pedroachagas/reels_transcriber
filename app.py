import streamlit as st
import instaloader
import numpy as np
import subprocess
import torch
import shutil
from IPython.display import display, Markdown
import whisper

st.title("Instagram Reel Transcription")

link = st.text_input("Enter Reel link here:")

if st.button("Download and Transcribe"):
    try:
        if link:
            Model = 'medium' 
            whisper_model = whisper.load_model(Model)

            if Model in whisper.available_models():
                st.write(f"**{Model} model is selected.**")
            else:
                st.write(f"**{Model} model is no longer available.**\n Please select one of the following:\n - {'\n - '.join(whisper.available_models())}")
                
            language = "English" 
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
            'Live transcription': True,
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

            reel = instaloader.Post.from_shortcode(instaloader.context, link.split("/")[-2])
            video_path_local = instaloader.context.download_video(reel, filename="temp", targetdir=None)
            
            video_transcription = whisper.transcribe(
                whisper_model,
                str(video_path_local),
                temperature=temperature,
                **args,
            )

            st.write(f"Reel downloaded successfully. Transcription: {video_transcription}")
            
        else:
            st.write("Please fill out the field")
            
    except Exception as e:
        st.write("Something went wrong. Please try again later.")
        st.write(e)
