
import os
import streamlit as st
import time
from gtts import gTTS
from pydub import AudioSegment
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import speech_recognition as sr
from PIL import Image
import base64
import re
from urllib.parse import urlparse

load_dotenv()

AudioSegment.converter = "C:/ffmpeg/ffmpeg/bin/ffmpeg.exe"
AudioSegment.ffmpeg = "C:/ffmpeg/ffmpeg/bin/ffmpeg.exe"
AudioSegment.ffprobe = "C:/ffmpeg/ffmpeg/bin/ffprobe.exe"

st.set_page_config(page_title="Article Analyzer 1.0", layout="wide")

st.markdown(
    """
    <style>
    .title {
        font-size: 3.5em;
        margin-top: -1.5em;
        margin-left: -0.6em;
        color: #1E90FF; /* Dodger Blue */
    }
    .version {
        font-size: 0.7em;
        vertical-align: super;
        color: cyan;
    }
    .subtitle {
        font-size: 1.5em;
        font-weight: bold;
        color: cyan;
    }
    .sidebar-section {
        margin-top: 1.5em;
    }
    .footer {
        position: fixed;
        right: 0;
        bottom: 0;
        padding: 10px 25px;
        font-size: 14px;
        color: #696969; /* Dim Gray */
    }
    .large-text-input input {
        font-size: 1.2em !important;
        padding: 0.5em 1em !important;
    }
    .sidebar-video {
        margin-top: -30px;
        margin-bottom: 20px;
        margin-left: -1rem;
        margin-right: -1rem;
        padding: 0;
    }
    .sidebar-video video {
        width: 100%;
        display: block;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

video_path = "Red and Blue Mascot Gaming Studio Free Logo1.mp4"  
if os.path.exists(video_path):
    video_html = f"""
        <div class = "sidebar-video">
            <video style="max-width: 100%; height: auto;" autoplay loop muted>
                <source src="data:video/mp4;base64,{get_base64_of_bin_file(video_path)}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
    """
    st.sidebar.markdown(video_html, unsafe_allow_html=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
    <script>
    const textInput = window.parent.document.querySelectorAll('input[data-testid="stTextInput"]')[0];
    textInput.classList.add('large-text-input');
    </script>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="title">Article 📃 Analyzer <span class="version">1.1</span></h1>', unsafe_allow_html=True)
st.markdown('<marquee scrollamount=16><h3 class="subtitle">Analyze and query multiple articles with ease</h3></marquee>', unsafe_allow_html=True)

st.sidebar.markdown("<hr>", unsafe_allow_html=True)

st.sidebar.markdown("<h1 style='font-size: 30px; font-weight: bold; color: cyan; text-align: center;'> Articles' URLs </h1>", unsafe_allow_html=True) 
st.sidebar.markdown("<hr class='sidebar-section'>", unsafe_allow_html=True)

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

if 'url_count' not in st.session_state:
    st.session_state.url_count = 1
    st.session_state.urls = [""]
    st.session_state.url_errors = [""]

def add_url():
    st.session_state.url_count += 1
    st.session_state.urls.append("")
    st.session_state.url_errors.append("")

def remove_url(index):
    st.session_state.urls.pop(index)
    st.session_state.url_errors.pop(index)
    st.session_state.url_count -= 1

def update_url(index):
    url = st.session_state.urls[index]
    if url and not is_valid_url(url):
        st.session_state.url_errors[index] = "Please enter a valid URL"
    else:
        st.session_state.url_errors[index] = ""

for i in range(st.session_state.url_count):
    col1, col2 = st.sidebar.columns([4, 1])
    with col1:
        url = st.sidebar.text_input(f"URL {i+1}", value=st.session_state.urls[i], placeholder="Enter URL", key=f"url_{i}", on_change=update_url, args=(i,))
        st.session_state.urls[i] = url
        if st.session_state.url_errors[i]:
            st.sidebar.error(st.session_state.url_errors[i])
    with col2:
        if st.sidebar.button("Remove", key=f"remove_{i}"):
            remove_url(i)
            st.sidebar.markdown("<hr>", unsafe_allow_html=True)
            st.experimental_rerun()

st.sidebar.markdown("<hr class='sidebar-section'>", unsafe_allow_html=True)

col1, col2 = st.sidebar.columns([1, 1])
with col1:
    add_url_button = st.button("Add URL", on_click=add_url)
with col2:
    process_url_clicked = st.button("Process URLs")

faiss_dir = "faiss_index"

main_placeholder = st.empty()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(openai_api_key=openai_api_key, temperature=0.9, max_tokens=500)

def any_urls_entered():
    return any(url.strip() for url in st.session_state.urls)

def any_valid_urls_entered():
    return any(url.strip() and is_valid_url(url) for url in st.session_state.urls)

if process_url_clicked:
    if any_urls_entered():
        if any_valid_urls_entered():
            loader = UnstructuredURLLoader(urls=st.session_state.urls)
            main_placeholder.markdown('Processing Initiated...')
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placeholder.markdown('Text Splitting Initiated...')
            time.sleep(1)
            docs = text_splitter.split_documents(data)

            temp_placeholder = st.empty()
            temp_placeholder.success('Text Splitting Completed...✅✅')
            time.sleep(2)  
            temp_placeholder.empty()

            embeddings = OpenAIEmbeddings()
            vectorstore_openai = FAISS.from_documents(docs, embeddings)
            main_placeholder.markdown('Embedding Vector Started Building...')
            time.sleep(1)

            if not os.path.exists(faiss_dir):
                os.makedirs(faiss_dir)
            vectorstore_openai.save_local(faiss_dir)

            final_placeholder = st.empty()
            final_placeholder.success('Vector Embeddings built and saved successfully...✅✅')
            time.sleep(2)  
            final_placeholder.empty()
            main_placeholder.markdown('You can now ask your query related to the article...') 
        else:
            st.error('Please enter a valid URL!')
    else:
        st.error('Please enter at least one URL before processing!')

def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.write("Say something...")
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text 
        except sr.UnknownValueError:
            return "Could not understand the audio"
        except sr.RequestError:
            return "Could not request results; check your network connection"
    except AttributeError:
        return "PyAudio is not installed. Please install PyAudio and try again."

st.markdown("---")
query = st.text_input("Enter your query:", label_visibility="collapsed", placeholder="Enter your query")

st.markdown("---") 

if st.button("🎤"):
    query = recognize_speech_from_mic()
    st.text_input("Enter your query:", value=query, key="query_input", label_visibility="collapsed")

st.markdown("---")

if query:
    if os.path.exists(faiss_dir):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        
        st.markdown("### Answer:")
        st.text_area("", value=result["answer"], height=150)
        
        tts = gTTS(result["answer"], lang='en-uk')
        tts.save("answer.mp3")
        
        audio = AudioSegment.from_mp3("answer.mp3")
        audio.export("answer.wav", format="wav")
        
        with open("answer.wav", "rb") as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')
        
        time.sleep(1)
        try:
            os.remove("answer.mp3")
            os.remove("answer.wav")
        except PermissionError:
            pass
        
        st.markdown("---")

        if st.button("Regenerate Answer"):
            st.experimental_rerun()

        sources = result.get("sources", "")
        if sources:
            st.markdown("<hr class='sidebar-section'>", unsafe_allow_html=True)
            st.markdown("### Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)

        st.markdown("---") 
    
st.markdown("""
    <style>
        .footer {
            position: fixed;
            right: 10px;
            bottom: 10px;
            padding: 10px 25px;
            font-size: 14px;
            color: #696969; /* Dim Gray */
        }
    </style>
    <div class="footer">
        &copy; 2024 <a href="https://www.linkedin.com/in/imstorm23203attherategmail/" target="_blank" style="color: cyan;">Arjun Singh Rajput</a>. All rights reserved.
    </div>
""", unsafe_allow_html=True) 