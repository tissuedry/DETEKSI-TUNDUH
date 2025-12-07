import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import av
import time
import os  # <--- WAJIB IMPORT INI UNTUK PATH
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Kantuk Cloud", layout="wide")
st.title("☁️ Deteksi Kantuk (Versi Cloud/WebRTC)")

# --- LOAD MODEL (DENGAN PATH ABSOLUT) ---
@st.cache_resource
def load_model_resources():
    # 1. Cari tahu folder tempat app.py berada
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Gabungkan folder tersebut dengan nama file
    path_model = os.path.join(base_path, 'best.pt')
    path_face = os.path.join(base_path, 'haarcascade_frontalface_default.xml')
    path_eye = os.path.join(base_path, 'haarcascade_eye.xml')

    try:
        # Cek manual apakah file ada (untuk debugging error)
        if not os.path.exists(path_model):
            st.error(f"❌ FILE HILANG: {path_model}")
            return None, None, None
            
        # Load Model dengan path lengkap
        model = YOLO(path_model)
        face_cascade = cv2.CascadeClassifier(path_face)
        eye_cascade = cv2.CascadeClassifier(path_eye)
        
        # Validasi load cascade (OpenCV tidak throw error, tapi return empty jika gagal)
        if face_cascade.empty() or eye_cascade.empty():
            st.error("❌ Gagal load XML Cascade. File mungkin corrupt.")
            return None, None, None
            
        return model, face_cascade, eye_cascade
        
    except Exception as e:
        st.error(f"❌ Error System: {e}")
        return None, None, None

model, face_cascade, eye_cascade = load_model_resources()

if model is None:
    st.warning("⚠️ Aplikasi berhenti karena model tidak ditemukan. Cek folder GitHub kamu!")
    st.stop()
else:
    st.success("✅ Model & Cascade berhasil dimuat!")

# --- PARAMETER USER ---
st.sidebar.title("Pengaturan")
CONFIDENCE_THRESHOLD = st.sidebar.slider("Threshold Keyakinan", 0.0, 1.0, 0.5, 0.05)
ALARM_THRESHOLD = st.sidebar.slider("Durasi Mata Tertutup (detik)", 0.5, 3.0, 1.0, 0.1)

# --- KELAS PROSESOR VIDEO (INTI LOGIKA) ---
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.closed_start_time = None

    def recv(self, frame):
        # Ambil gambar dari WebRTC
        img = frame.to_ndarray(format="bgr24")
        
        # Proses Gambar
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Deteksi Wajah
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        any_eye_closed = False
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
            for (ex, ey, ew, eh) in eyes:
                eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
                try:
                    # YOLO Logic
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
                        label = "Tutup" if is_closed else "Buka"
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)
                        cv2.putText(roi_color, label, (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                except:
                    pass

        # Logika Waktu & Alarm Visual
        now = time.time()
        status_text = "AMAN"
        color_text = (0, 255, 0)

        if len(faces) > 0:
            if any_eye_closed:
                if self.closed_start_time is None:
                    self.closed_start_time = now
                
                duration = now - self.closed_start_time
                if duration >= ALARM_THRESHOLD:
                    status_text = f"!!! MENGANTUK ({duration:.1f}s) !!!"
                    color_text = (0, 0, 255)
                    # Bingkai Merah Tebal
                    cv2.rectangle(img, (0,0), (img.shape[1], img.shape[0]), (0,0,255), 20)
                else:
                    status_text = f"Mata Tertutup ({duration:.1f}s)"
                    color_text = (0, 165, 255)
            else:
                self.closed_start_time = None
        else:
            self.closed_start_time = None

        cv2.putText(img, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_text, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI UTAMA WEB RTC ---
st.write("Klik tombol **START** di bawah. Izinkan akses kamera jika diminta browser.")

rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="drowsiness-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    video_processor_factory=DrowsinessProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)