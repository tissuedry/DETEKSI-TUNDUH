import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time
from datetime import datetime
import base64  # <--- Ganti pygame dengan base64

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Kantuk Driver", layout="wide")

st.title("🚗 Sistem Deteksi Kantuk Pengemudi")
st.write("Versi Cloud (Audio via Browser)")

# --- FUNGSI AUDIO BROWSER ---
def get_audio_html(file_path):
    # Ubah file audio jadi kode base64 agar bisa diputar browser
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        # HTML hidden audio player yang autoplay & loop
        return f"""
            <audio autoplay loop>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
    except Exception as e:
        return ""

# --- SIDEBAR & SETUP ---
st.sidebar.title("Pengaturan")
confidence_threshold = st.sidebar.slider("Threshold Keyakinan Model", 0.0, 1.0, 0.5, 0.05)
alarm_threshold = st.sidebar.slider("Durasi Mata Tertutup (detik)", 0.5, 3.0, 1.0, 0.1)

# File audio harus ada di folder github
audio_file = "alarm.mp3" 

# --- LOAD MODEL & RESOURCE ---
@st.cache_resource
def load_model():
    return YOLO('best.pt') 

@st.cache_resource
def load_cascades():
    face_c = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_c = cv2.CascadeClassifier('haarcascade_eye.xml')
    return face_c, eye_c

try:
    model = load_model()
    face_cascade, eye_cascade = load_cascades()
    st.sidebar.success("Model dimuat!")
except Exception as e:
    st.error("Error memuat model/xml.")
    st.stop()

# --- STATE MANAGEMENT ---
if 'history_log' not in st.session_state:
    st.session_state.history_log = []
if 'closed_start_time' not in st.session_state:
    st.session_state.closed_start_time = None

# --- UI UTAMA ---
col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("Status & Log")
    status_placeholder = st.empty()
    log_placeholder = st.empty()
    # Placeholder Khusus Suara (Invisible)
    audio_placeholder = st.empty() 
    
    run = st.checkbox('Aktifkan Kamera', value=False)

with col1:
    st.subheader("Live Feed Kamera")
    frame_placeholder = st.empty()

# --- LOGIKA UTAMA ---
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Gagal mengakses kamera.")
        break

    frame = cv2.flip(frame, 1)
    # Resize frame biar ringan di cloud
    frame = cv2.resize(frame, (640, 480))
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    current_status = "Menunggu Wajah..."
    color_status = "blue"
    any_eye_closed = False
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
        
        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
            try:
                eye_pil = Image.fromarray(cv2.cvtColor(eye_roi, cv2.COLOR_BGR2RGB))
                results = model(eye_pil, verbose=False)
                probs = results[0].probs
                conf = probs.top1conf.item()
                pred_class = results[0].names[probs.top1]

                if conf > confidence_threshold:
                    is_closed = (pred_class == 'closed')
                    if is_closed:
                        any_eye_closed = True
                    
                    color = (0, 0, 255) if is_closed else (0, 255, 0)
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)
            except:
                pass

    # --- LOGIKA KANTUK & AUDIO BROWSER ---
    now = datetime.now()
    play_sound = False # Flag untuk memicu suara
    
    if len(faces) > 0:
        if any_eye_closed:
            if st.session_state.closed_start_time is None:
                st.session_state.closed_start_time = now
            
            duration = (now - st.session_state.closed_start_time).total_seconds()
            
            if duration >= alarm_threshold:
                current_status = f"⚠️ MENGANTUK! ({duration:.1f}s)"
                color_status = "red"
                play_sound = True # Trigger suara
                
                log_entry = (now.strftime("%H:%M:%S"), "MENGANTUK!")
                if not st.session_state.history_log or st.session_state.history_log[-1][1] != "MENGANTUK!":
                    st.session_state.history_log.append(log_entry)
            else:
                current_status = f"Mata Tertutup ({duration:.1f}s)"
                color_status = "orange"
        else:
            st.session_state.closed_start_time = None
            current_status = "AMAN"
            color_status = "green"
            
            log_entry = (now.strftime("%H:%M:%S"), "AMAN")
            if not st.session_state.history_log or st.session_state.history_log[-1][1] == "MENGANTUK!":
                 st.session_state.history_log.append(log_entry)
    
    # Update UI
    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
    
    # UPDATE STATUS & AUDIO
    status_html = f"""
    <div style='padding: 10px; border-radius: 5px; background-color: {color_status}; color: white; text-align: center;'>
        <h3>{current_status}</h3>
    </div>
    """
    status_placeholder.markdown(status_html, unsafe_allow_html=True)

    # LOGIKA PEMUTAR SUARA YANG BARU
    if play_sound:
        # Masukkan HTML Audio ke placeholder
        audio_code = get_audio_html(audio_file)
        audio_placeholder.markdown(audio_code, unsafe_allow_html=True)
    else:
        # Kosongkan placeholder (Stop suara)
        audio_placeholder.empty()

    time.sleep(0.01)

cap.release()