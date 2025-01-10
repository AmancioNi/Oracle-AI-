import os
import cv2
import yt_dlp
import asyncio
import pyaudio
import streamlit as st
from deepface import DeepFace
from moviepy import VideoFileClip, TextClip, CompositeVideoClip
import matplotlib.pyplot as plt
import whisper
import logging

# Configuração do Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuração do Streamlit
st.set_page_config(page_title="Transcrição e Análise de Vídeos", layout="wide", initial_sidebar_state="expanded")

# Inicialização de estado do Streamlit
if 'video_path' not in st.session_state:
    st.session_state['video_path'] = None

# Função para baixar vídeos do YouTube
@st.cache_data
def download_youtube_video(url, output_dir="downloads"):
    ydl_opts = {
        "format": "best",
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegVideoConvertor", "preferedformat": "mp4"}],
    }
    os.makedirs(output_dir, exist_ok=True)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info_dict)

# Função para processar vídeo e inserir análise de emoções
def analyze_emotions(video_path, output_path="processed_video.mp4"):
    video_capture = cv2.VideoCapture(video_path)
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    output_video = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    st.info("Processando vídeo para análise de emoções...")
    progress_bar = st.progress(0)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        try:
            analysis = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
            frame = draw_emotions_on_frame(frame, analysis if isinstance(analysis, list) else [analysis])
        except Exception as e:
            logger.warning(f"Erro ao processar o frame {frame_number}: {e}")
        output_video.write(frame)
        frame_number += 1
        progress_bar.progress(frame_number / frame_count)

    video_capture.release()
    output_video.release()
    progress_bar.empty()
    st.success(f"Processamento do vídeo concluído! Salvo como {output_path}")
    return output_path

# Mapeamento de emoções para cores
EMOTIONS_COLORS = {
    "happy": (0, 255, 0),       # Verde
    "sad": (255, 0, 0),         # Azul
    "angry": (0, 0, 255),       # Vermelho
    "surprise": (255, 255, 0),  # Amarelo
    "fear": (128, 0, 128),      # Roxo
    "neutral": (255, 255, 255), # Branco
    "disgust": (0, 128, 0),     # Verde escuro
    "Unknown": (128, 128, 128), # Cinza
}

# Função para desenhar emoções no frame do vídeo
def draw_emotions_on_frame(frame, analysis):
    for face_data in analysis:
        if "region" in face_data:
            x, y, w, h = face_data["region"].get("x", 0), face_data["region"].get("y", 0), face_data["region"].get("w", 0), face_data["region"].get("h", 0)
            emotion = face_data.get("dominant_emotion", "Unknown")
            color = EMOTIONS_COLORS.get(emotion, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame

# Função para gerar contexto a partir das emoções detectadas
def generate_context_from_emotions(emotion_data):
    context = "As emoções detectadas indicam os seguintes padrões:\n"
    emotion_counts = {}
    for face_data in emotion_data:
        emotion = face_data.get("dominant_emotion", "Unknown")
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    if not emotion_counts:
        context += "Nenhuma emoção foi detectada.\n"
    else:
        for emotion, count in emotion_counts.items():
            context += f"- A emoção '{emotion}' foi detectada {count} vez(es).\n"

    context += "Esses dados podem ser usados para entender os padrões emocionais do vídeo."
    return context

# Função para plotar gráfico das emoções
def plot_emotion_distribution(emotion_data):
    emotion_counts = {}
    for face_data in emotion_data:
        emotion = face_data.get("dominant_emotion", "Unknown")
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(emotions, counts, color=[EMOTIONS_COLORS.get(e, "gray") for e in emotions])
    plt.title("Distribuição de Emoções")
    plt.xlabel("Emoções")
    plt.ylabel("Frequência")
    plt.tight_layout()
    st.pyplot(plt)

# Função para transcrever áudio usando Whisper
def transcribe_audio_dynamic(video_filename):
    try:
        model = whisper.load_model("base")
        st.info("Iniciando transcrição...")
        result = model.transcribe(video_filename, verbose=False)

        dynamic_transcription = ""
        for segment in result['segments']:
            dynamic_transcription += f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}\n"
        return dynamic_transcription
    except Exception as e:
        st.error(f"Erro durante a transcrição: {e}")
        logger.error(f"Erro na transcrição: {e}")
        raise
# Interface principal
st.title("AI SpeakSense - Transcrição e Análise de Vídeos")

# Layout com colunas
col1, col2, col3 = st.columns([1, 1, 1])

# Coluna 1: Vídeo
with col1:
    st.markdown(":film_projector: **Vídeo**")
    video_path = st.session_state.get("video_path")

    if not video_path:
        youtube_url = st.text_input("URL do YouTube", placeholder="Cole a URL do vídeo")
        uploaded_file = st.file_uploader("Ou carregue um arquivo de vídeo", type=["mp4", "avi", "mkv"])

        if st.button("Carregar Vídeo"):
            if youtube_url:
                video_path = download_youtube_video(youtube_url)
                st.session_state["video_path"] = video_path
                st.success("Vídeo baixado com sucesso!")
            elif uploaded_file:
                video_path = os.path.join("uploads", uploaded_file.name)
                os.makedirs("uploads", exist_ok=True)
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.read())
                st.session_state["video_path"] = video_path
                st.success("Arquivo carregado com sucesso!")
            else:
                st.error("Insira uma URL ou carregue um arquivo de vídeo.")
    else:
        st.video(video_path)

# Coluna 2: Transcrição
with col2:
    st.markdown(":keyboard: **Transcrição**")
    if video_path and st.button("Iniciar Transcrição"):
        transcribed_text = transcribe_audio_dynamic(video_path)
        st.text_area("Transcrição Completa", transcribed_text, height=300)

# Coluna 3: Análise de Emoções
with col3:
    st.markdown(":sparkles: **Análise de Emoções**")
    if video_path and st.button("Processar Análise"):
        processed_video_path = analyze_emotions(video_path)
        st.video(processed_video_path)
