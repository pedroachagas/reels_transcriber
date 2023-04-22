import streamlit as st
import whisper
import warnings
import openai
from pytube import YouTube
import instaloader
import streamlit.components.v1 as components


# Constants
USD_TO_BRL = 5.0  # The conversion rate from USD to BRL
API_COST_PER_TOKEN = 0.002 / 1000  # The OpenAI API cost per token

openai.api_key = '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'

def tokens_to_brl(tokens):
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
warnings.filterwarnings('ignore')

st.title("Video Transcription App")

link = st.text_input("Enter Instagram Reel or YouTube video link here:")

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
            with st.spinner('Getting video link...'):
                video_url, platform = get_video_url(link)
                if not video_url:
                    st.write("Invalid link. Please provide a valid Instagram Reel or YouTube video link.")
                    raise ValueError("Invalid link")

            # Transcribe video
            with st.spinner('Transcribing video...'):
                transcription = whisper.transcribe(
                    whisper_model,
                    video_url,
                    temperature=temperature,
                    **args,
                    )['text']

            if len(transcription.split()) > 2000:
                st.warning("The transcription is longer than 2000 words. Skipping the processing step and returning the unprocessed transcription.")
                st.session_state.transcription_processed = transcription
                st.session_state.title = 'Raw transcription'
            else:
                with st.spinner('Processing transcription...'):
                    title, title_cost = create_title(transcription)
                    transcription_processed, process_cost = process_transcription(transcription)
                    transcription_processed = transcription_processed.replace('#', '##')

                st.session_state.transcription_processed = transcription_processed
                st.session_state.title = title
                st.session_state.title_cost = title_cost
                st.session_state.process_cost = process_cost
                st.session_state.transcription = transcription
                st.success('Transcription completed!')

        else:
            st.write("Please fill out the field")

    except Exception as e:
        st.write("Something went wrong. Please try again later.")
        st.write(e)

if "transcription_processed" in st.session_state:
    with st.expander("See transcription"):
        st.markdown("# " + st.session_state.title)
        st.markdown(st.session_state.transcription_processed)
        copy_button(st.session_state.transcription_processed)
        st.caption(f"Cost in BRL for processing transcription: R$ {st.session_state.title_cost + st.session_state.process_cost:.4f}")

    # New feature: user inputs a text prompt
    prompt = st.text_input("Enter a text prompt to modify the processed transcription:")

    if st.button("Apply Prompt"):
        with st.spinner("Applying prompt..."):
            modified_transcription, prompt_cost = apply_prompt(prompt, st.session_state.transcription_processed)
        st.session_state.modified_transcription = modified_transcription
        st.session_state.prompt_cost = prompt_cost

    if "modified_transcription" in st.session_state:
        with st.expander("See modified transcription"):
            st.markdown(st.session_state.modified_transcription)
            copy_button(st.session_state.modified_transcription)
            st.caption(f"Cost in BRL for applying prompt: R$ {st.session_state.prompt_cost:.4f}")

    if st.button("Retry"):
        with st.spinner('Reprocessing transcription...'):
            title = create_title(st.session_state.transcription)
            transcription_processed = process_transcription(st.session_state.transcription, 0.7).replace('#', '##')
        st.session_state.transcription_processed_reprocessed = transcription_processed
        st.session_state.title_reprocessed = title

    if "transcription_processed_reprocessed" in st.session_state:
        with st.expander("See reprocessed transcription"):
            st.markdown("# " + st.session_state.title_reprocessed)
            st.markdown(st.session_state.transcription_processed_reprocessed)