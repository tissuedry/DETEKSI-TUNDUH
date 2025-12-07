import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time
from datetime import datetime
import pygame  # <--- LIBRARY AUDIO BARU

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Kantuk Driver", layout="wide")

st.title("🚗 Sistem Deteksi Kantuk Pengemudi")
st.write("Menggunakan YOLOv8 + Haar Cascade + Custom Audio")

# --- INITIALIZE PYGAME MIXER (AUDIO) ---
try:
    pygame.mixer.init()
except:
    pass # Cegah error jika init dipanggil berulang

# --- SIDEBAR & SETUP ---
st.sidebar.title("Pengaturan")
confidence_threshold = st.sidebar.slider("Threshold Keyakinan Model", 0.0, 1.0, 0.5, 0.05)
alarm_threshold = st.sidebar.slider("Durasi Mata Tertutup (detik)", 0.5, 3.0, 1.0, 0.1)

# Pilihan File Audio (Opsional, pastikan file ada di folder)
audio_file = "alarm.mp3" # <--- GANTI NAMA FILE DISINI JIKA BEDA

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
    st.sidebar.success("Model & Cascade berhasil dimuat!")
except Exception as e:
    st.error(f"Error memuat file: {e}. Pastikan 'best.pt' dan file XML ada.")
    st.stop()

# --- STATE MANAGEMENT ---
if 'history_log' not in st.session_state:
    st.session_state.history_log = []
if 'closed_start_time' not in st.session_state:
    st.session_state.closed_start_time = None
if 'alarm_active' not in st.session_state:
    st.session_state.alarm_active = False

# --- UI UTAMA ---
col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("Status & Log")
    status_placeholder = st.empty()
    log_placeholder = st.empty()
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
                    label_text = f"{'Tutup' if is_closed else 'Buka'} {conf:.2f}"
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)
                    cv2.putText(roi_color, label_text, (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            except:
                pass

    # --- LOGIKA KANTUK & ALARM MP3 ---
    now = datetime.now()
    
    if len(faces) > 0:
        if any_eye_closed:
            if st.session_state.closed_start_time is None:
                st.session_state.closed_start_time = now
            
            duration = (now - st.session_state.closed_start_time).total_seconds()
            
            # === KONDISI MENGANTUK ===
            if duration >= alarm_threshold:
                current_status = f"⚠️ MENGANTUK! ({duration:.1f}s)"
                color_status = "red"
                
                # >> BUNYIKAN MP3 <<
                if not st.session_state.alarm_active:
                    try:
                        # Load file MP3
                        pygame.mixer.music.load(audio_file)
                        
                        # SET VOLUME MAKSIMAL (Range 0.0 sampai 1.0)
                        pygame.mixer.music.set_volume(1.0)  # <--- TAMBAHKAN BARIS INI
                        
                        # Play loop (-1 artinya looping terus)
                        pygame.mixer.music.play(-1)
                        st.session_state.alarm_active = True
                    except Exception as e:
                        st.error(f"Gagal memutar audio: {e}. Cek nama file!")

                # Log
                log_entry = (now.strftime("%H:%M:%S"), "MENGANTUK!")
                if not st.session_state.history_log or st.session_state.history_log[-1][1] != "MENGANTUK!":
                    st.session_state.history_log.append(log_entry)
            else:
                current_status = f"Mata Tertutup ({duration:.1f}s)"
                color_status = "orange"
        else:
            # === KONDISI AMAN (MATA TERBUKA) ===
            st.session_state.closed_start_time = None
            current_status = "AMAN (Mata Terbuka)"
            color_status = "green"
            
            # Stop Alarm MP3
            if st.session_state.alarm_active:
                pygame.mixer.music.stop()
                st.session_state.alarm_active = False
            
            log_entry = (now.strftime("%H:%M:%S"), "AMAN")
            if not st.session_state.history_log or st.session_state.history_log[-1][1] == "MENGANTUK!":
                 st.session_state.history_log.append(log_entry)
    else:
        # Jika wajah hilang, matikan alarm
        if st.session_state.alarm_active:
             pygame.mixer.music.stop()
             st.session_state.alarm_active = False

    # Update UI
    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
    
    status_html = f"""
    <div style='padding: 10px; border-radius: 5px; background-color: {color_status}; color: white; text-align: center;'>
        <h3>{current_status}</h3>
    </div>
    """
    status_placeholder.markdown(status_html, unsafe_allow_html=True)

    log_text = "**Riwayat:**\n\n"
    for time_str, status in reversed(st.session_state.history_log[-5:]):
        log_text += f"- {time_str}: {status}\n"
    log_placeholder.markdown(log_text)

    time.sleep(0.01)

# Cleanup
cap.release()
pygame.mixer.music.stop() # Matikan suara saat exit