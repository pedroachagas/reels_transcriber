import streamlit as st
import whisper
from package import *

st.title("Video Transcription App")

link = st.text_input("Enter Instagram Reel or YouTube video link here:")
uploaded_file = st.file_uploader("Or upload a video file:", type=["mp4"])

selected_model, selected_language_code, process_transcription_toggle = display_sidebar()
set_api_key()

if st.button("Transcribe") and (link or uploaded_file):
    # Download video
    if uploaded_file:
        video_url = save_uploaded_file(uploaded_file)
        platform = "uploaded"
    else:
        with st.spinner('Getting video link...'):
            video_url, platform = get_video_url(link)
            if not video_url:
                st.write("Invalid link. Please provide a valid Instagram Reel or YouTube video link.")
                raise ValueError("Invalid link")

    @st.cache_data(show_spinner=False)
    def transcribe(video_url, selected_model, selected_language_code, **args):
        whisper_model = whisper.load_model(selected_model)
        transcription = whisper.transcribe(
            whisper_model,
            video_url,
            temperature=0.15,
            language=selected_language_code,
            verbose=None,
            task='transcribe',
            best_of=5,
            beam_size=8,
            patience=1.0,
            length_penalty=None,
            suppress_tokens="-1",
            initial_prompt=None,
            condition_on_previous_text=True,
            fp16=False,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            **args
        )['text']
        return transcription

    with st.spinner('Transcribing video...'):
        transcription = transcribe(video_url, selected_model, selected_language_code)

    if process_transcription_toggle and len(transcription.split()) > 2500:
        st.warning("The transcription is longer than 2500 words. Skipping the processing step and returning the unprocessed transcription.")
        transcription_processed = None
        process_cost = 0
        title_cost = 0
        title = 'Raw transcription'
    elif not process_transcription_toggle:
        transcription_processed = None
        process_cost = 0
        title_cost = 0
        title = 'Raw transcription'   
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

if ('transcription_processed' in st.session_state and st.session_state.transcription_processed is not None) and process_transcription_toggle:
    show_original = st.checkbox("Show raw transcription", value=False)

    with st.expander("See transcription"):
        if show_original:
            st.markdown("# Raw transcription")
            st.markdown(st.session_state.transcription)
            copy_button(st.session_state.transcription)
            st.caption(f"Cost in BRL for processing transcription: R$ {st.session_state.title_cost + st.session_state.process_cost:.4f}")
        else:
            st.markdown("# " + st.session_state.title)
            st.markdown(st.session_state.transcription_processed)
            copy_button(st.session_state.transcription_processed)
            st.caption(f"Cost in BRL for processing transcription: R$ {st.session_state.title_cost + st.session_state.process_cost:.4f}")

            if st.button("Retry"):
                with st.spinner('Reprocessing transcription...'):
                    title, title_cost = create_title(st.session_state.transcription)
                    transcription_processed, prompt_cost = process_transcription(st.session_state.transcription, 0.7)
                st.session_state.transcription_processed_reprocessed = transcription_processed.replace('#', '##')
                st.session_state.title_reprocessed = title
                st.session_state.prompt_cost = prompt_cost
                st.session_state.title_cost = title_cost

    with st.expander("See reprocessed transcription"):
        if "transcription_processed_reprocessed" in st.session_state:
            st.markdown("# " + st.session_state.title_reprocessed)
            st.markdown(st.session_state.transcription_processed_reprocessed)
            copy_button(st.session_state.transcription_processed)
            st.caption(f"Cost in BRL for applying prompt: R$ {st.session_state.prompt_cost + st.session_state.title_cost:.4f}")

if 'transcription_processed' in st.session_state and st.session_state.transcription_processed is not None:
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

elif not process_transcription_toggle and "transcription" in st.session_state:
    with st.expander("See transcription"):
        st.markdown("# " + st.session_state.title)
        st.markdown(st.session_state.transcription)
        copy_button(st.session_state.transcription)

    if st.button("Process transcription"):
        with st.spinner('Processing transcription...'):
            title, title_cost = create_title(st.session_state.transcription)
            transcription_processed, prompt_cost = process_transcription(st.session_state.transcription, 0.7)
        st.session_state.transcription_processed_from_raw = transcription_processed.replace('#', '##')
        st.session_state.title_from_raw = title
        st.session_state.prompt_cost = prompt_cost
        st.session_state.title_cost = title_cost

    if "transcription_processed_from_raw" in st.session_state:    
        with st.expander("See processed transcription"):
            st.markdown("# " + st.session_state.title_from_raw)
            st.markdown(st.session_state.transcription_processed_from_raw)
            st.caption(f"Cost in BRL for processing transcription: R$ {st.session_state.prompt_cost + st.session_state.title_cost:.4f}")

        prompt = st.text_input("Enter a text prompt to modify the processed transcription:")
 
        if st.button("Apply Prompt"):
            with st.spinner("Applying prompt..."):
                modified_transcription, prompt_cost = apply_prompt(prompt, st.session_state.transcription_processed_from_raw)
            st.session_state.modified_transcription = modified_transcription
            st.session_state.prompt_cost = prompt_cost

        if "modified_transcription" in st.session_state:
            with st.expander("See modified transcription"):
                st.markdown(st.session_state.modified_transcription)
                copy_button(st.session_state.modified_transcription)
                st.caption(f"Cost in BRL for applying prompt: R$ {st.session_state.prompt_cost:.4f}")

