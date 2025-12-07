import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import av
import time
import os
import queue
import base64
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Kantuk Cloud + Audio", layout="wide")
st.title("☁️ Deteksi Kantuk (WebRTC + Audio Alarm)")

# --- FUNGSI AUDIO HTML (Jalan di Browser) ---
def get_audio_html(file_path):
    try:
        # Baca file MP3 dan ubah jadi teks (Base64)
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        # Autoplay, Loop, dan Hidden
        return f"""
            <audio autoplay loop>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
    except Exception as e:
        return ""

# --- LOAD MODEL & INITIAL SETUP ---
@st.cache_resource
def load_resources():
    base_path = os.path.dirname(os.path.abspath(__file__))
    path_model = os.path.join(base_path, 'best.pt')
    path_face = os.path.join(base_path, 'haarcascade_frontalface_default.xml')
    path_eye = os.path.join(base_path, 'haarcascade_eye.xml')
    path_audio = os.path.join(base_path, 'alarm.mp3') # Pastikan file ini ada!

    model = None
    face_c = None
    eye_c = None
    
    try:
        model = YOLO(path_model)
        face_c = cv2.CascadeClassifier(path_face)
        eye_c = cv2.CascadeClassifier(path_eye)
    except Exception as e:
        st.error(f"Error Load: {e}")
    
    return model, face_c, eye_c, path_audio

model, face_cascade, eye_cascade, audio_path = load_resources()

# Antrian untuk komunikasi antara Video Processor dan UI Utama
# Ini adalah "Kotak Surat" tempat Processor mengirim status kantuk
result_queue = queue.Queue()

# --- SETTING SIDEBAR ---
st.sidebar.title("Pengaturan")
CONFIDENCE_THRESHOLD = st.sidebar.slider("Threshold Keyakinan", 0.0, 1.0, 0.5, 0.05)
ALARM_THRESHOLD = st.sidebar.slider("Durasi Mata Tertutup (detik)", 0.5, 3.0, 1.0, 0.1)

# --- PROSESOR VIDEO ---
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.closed_start_time = None
        self.alarm_on = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Proses Flip & Gray
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        any_eye_closed = False
        
        # --- DETEKSI MATA & YOLO ---
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
            for (ex, ey, ew, eh) in eyes:
                eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
                try:
                    eye_pil = Image.fromarray(cv2.cvtColor(eye_roi, cv2.COLOR_BGR2RGB))
                    results = model(eye_pil, verbose=False)
                    probs = results[0].probs
                    conf = probs.top1conf.item()
                    pred_class = results[0].names[probs.top1]

                    if conf > CONFIDENCE_THRESHOLD:
                        is_closed = (pred_class == 'closed')
                        if is_closed:
                            any_eye_closed = True
                        
                        color = (0, 0, 255) if is_closed else (0, 255, 0)
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)
                except:
                    pass

        # --- LOGIKA ALARM ---
        now = time.time()
        is_drowsy = False

        if len(faces) > 0:
            if any_eye_closed:
                if self.closed_start_time is None:
                    self.closed_start_time = now
                
                duration = now - self.closed_start_time
                if duration >= ALARM_THRESHOLD:
                    is_drowsy = True # <--- KETEMU KANTUK
                    cv2.rectangle(img, (0,0), (img.shape[1], img.shape[0]), (0,0,255), 20)
                    cv2.putText(img, "!!! MENGANTUK !!!", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
            else:
                self.closed_start_time = None
        else:
            self.closed_start_time = None

        # --- KIRIM SINYAL KE UI UTAMA ---
        # Kita masukkan status 'is_drowsy' ke dalam kotak surat (queue)
        # Main thread nanti akan membacanya
        try:
            result_queue.put_nowait(is_drowsy)
        except queue.Full:
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI UTAMA ---
col1, col2 = st.columns([3, 1])

with col2:
    st.write("**Status Audio:**")
    audio_placeholder = st.empty() # Tempat menaruh pemutar musik tersembunyi

with col1:
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # Simpan konteks webrtc ke variabel 'ctx'
    ctx = webrtc_streamer(
        key="drowsiness-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=DrowsinessProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# --- LOOP PENGHUBUNG (THE BRIDGE) ---
# Kode di bawah ini bertugas membaca Queue dan menyalakan Audio di Browser
if ctx.state.playing:
    while True:
        try:
            # 1. Cek apakah ada pesan baru dari Video Processor (Tunggu max 1 detik)
            result = result_queue.get(timeout=1.0)
            
            # 2. Jika Pesan = True (Mengantuk), mainkan lagu
            if result:
                audio_html = get_audio_html(audio_path)
                audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
            else:
                # Jika False (Aman), matikan lagu (kosongkan placeholder)
                audio_placeholder.empty()
                
        except queue.Empty:
            # Jika tidak ada pesan (misal video lagi loading), lanjut saja
            continue
        except Exception as e:
            break