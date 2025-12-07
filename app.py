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
import gdown  # Pastikan library ini ada di requirements.txt
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Kantuk Cloud + Audio", layout="wide")
st.title("☁️ Deteksi Kantuk (WebRTC + Audio Alarm)")

# --- FUNGSI AUDIO HTML (Jalan di Browser) ---
def get_audio_html(file_path):
    try:
        # Baca file MP3 dan ubah jadi teks (Base64) agar bisa diputar browser
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        # Autoplay, Loop, dan Hidden (tidak terlihat user)
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
    
    # Definisi Path File
    path_model = os.path.join(base_path, 'best.pt')
    path_face = os.path.join(base_path, 'haarcascade_frontalface_default.xml')
    path_eye = os.path.join(base_path, 'haarcascade_eye.xml')
    path_audio = os.path.join(base_path, 'alarm.mp3') # Sesuai request: alarm.mp3

    # --- LOGIKA DOWNLOAD OTOMATIS (Fix Error 'Ran out of input') ---
    # ID file diambil dari link Google Drive Anda: 1VHqQVyeFEI3Dghm7vSxUl8BLwrq1iBEh
    file_id = '1VHqQVyeFEI3Dghm7vSxUl8BLwrq1iBEh'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    # Cek apakah file model ada dan ukurannya valid (lebih dari 10KB)
    # Jika tidak ada atau ukurannya kecil (pointer file GitHub), download ulang
    if not os.path.exists(path_model) or os.path.getsize(path_model) < 10000:
        with st.spinner("Sedang mengunduh model dari Google Drive... (Harap tunggu, proses ini hanya sekali)"):
            try:
                # Hapus file lama jika ada tapi rusak
                if os.path.exists(path_model):
                    os.remove(path_model)
                gdown.download(url, path_model, quiet=False)
                st.success("Model berhasil diunduh!")
            except Exception as e:
                st.error(f"Gagal mengunduh model: {e}")
                return None, None, None, None
    # -------------------------------------------------------------

    model = None
    face_c = None
    eye_c = None
    
    try:
        model = YOLO(path_model)
        face_c = cv2.CascadeClassifier(path_face)
        eye_c = cv2.CascadeClassifier(path_eye)
    except Exception as e:
        st.error(f"Error Load Resources: {e}")
    
    return model, face_c, eye_c, path_audio

# Load semua resource ke variabel global
model, face_cascade, eye_cascade, audio_path = load_resources()

# Antrian untuk komunikasi antara Video Processor dan UI Utama
# Ini adalah "Kotak Surat" tempat Processor mengirim status kantuk
result_queue = queue.Queue()

# --- SETTING SIDEBAR ---
st.sidebar.title("Pengaturan")
CONFIDENCE_THRESHOLD = st.sidebar.slider("Threshold Keyakinan (YOLO)", 0.0, 1.0, 0.5, 0.05)
ALARM_THRESHOLD = st.sidebar.slider("Durasi Mata Tertutup (detik)", 0.5, 3.0, 1.0, 0.1)

# --- PROSESOR VIDEO (Backend WebRTC) ---
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.closed_start_time = None
        self.alarm_on = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Proses Flip & Gray
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Pastikan cascade classifier berhasil dimuat
        if face_cascade is None or eye_cascade is None or model is None:
             cv2.putText(img, "Model Loading Error...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
             return av.VideoFrame.from_ndarray(img, format="bgr24")

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        any_eye_closed = False
        
        # --- DETEKSI WAJAH ---
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            # --- DETEKSI MATA ---
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
            for (ex, ey, ew, eh) in eyes:
                eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
                try:
                    # Konversi ke PIL Image untuk YOLO
                    eye_pil = Image.fromarray(cv2.cvtColor(eye_roi, cv2.COLOR_BGR2RGB))
                    
                    # Prediksi YOLO
                    results = model(eye_pil, verbose=False)
                    probs = results[0].probs
                    conf = probs.top1conf.item()
                    pred_class = results[0].names[probs.top1]

                    # Filter berdasarkan threshold keyakinan user
                    if conf > CONFIDENCE_THRESHOLD:
                        is_closed = (pred_class == 'closed')
                        if is_closed:
                            any_eye_closed = True
                        
                        # Gambar kotak di mata (Merah=Tutup, Hijau=Buka)
                        color = (0, 0, 255) if is_closed else (0, 255, 0)
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)
                        
                        # Tulis status di atas mata
                        label = f"{pred_class} {conf:.2f}"
                        cv2.putText(roi_color, label, (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                except Exception as e:
                    pass

        # --- LOGIKA ALARM (TIMER) ---
        now = time.time()
        is_drowsy = False

        if len(faces) > 0:
            if any_eye_closed:
                if self.closed_start_time is None:
                    self.closed_start_time = now
                
                # Hitung durasi mata tertutup
                duration = now - self.closed_start_time
                if duration >= ALARM_THRESHOLD:
                    is_drowsy = True # <--- KETEMU KANTUK
                    
                    # Tampilkan peringatan visual di layar
                    cv2.rectangle(img, (0,0), (img.shape[1], img.shape[0]), (0,0,255), 20)
                    cv2.putText(img, "!!! MENGANTUK !!!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
            else:
                self.closed_start_time = None
        else:
            self.closed_start_time = None

        # --- KIRIM SINYAL KE UI UTAMA ---
        # Masukkan status 'is_drowsy' ke queue agar UI thread bisa membacanya dan memutar audio
        try:
            result_queue.put_nowait(is_drowsy)
        except queue.Full:
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI UTAMA (FRONTEND) ---
col1, col2 = st.columns([3, 1])

with col2:
    st.write("### Status Audio")
    audio_placeholder = st.empty() # Placeholder untuk menyuntikkan HTML audio

with col1:
    # Konfigurasi WebRTC (STUN Server Google agar stabil di Cloud)
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # Inisialisasi WebRTC Streamer
    ctx = webrtc_streamer(
        key="drowsiness-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=DrowsinessProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# --- LOOP PENGHUBUNG (THE BRIDGE) ---
# Kode ini terus berjalan selama streamer aktif untuk memutar audio
if ctx.state.playing:
    while True:
        try:
            # 1. Ambil status dari Video Processor
            result = result_queue.get(timeout=1.0)
            
            # 2. Jika Mengantuk (True), mainkan lagu
            if result:
                if audio_path and os.path.exists(audio_path):
                    audio_html = get_audio_html(audio_path)
                    audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
                else:
                    audio_placeholder.error("File alarm.mp3 tidak ditemukan!")
            else:
                # Jika Aman (False), hentikan/kosongkan audio
                audio_placeholder.empty()
                
        except queue.Empty:
            continue
        except Exception as e:
            break