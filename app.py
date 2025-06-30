"""
YOLO BoTSORT Tracker - Streamlit Web Arayüzü
Kamera ve video dosyası ile nesne takibi
"""

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
from pathlib import Path
import time
from PIL import Image
import io

# Sayfa yapılandırması
st.set_page_config(
    page_title="YOLO BoTSORT Tracker",
    page_icon="🎯",
    layout="wide"
)

# CSS stilleri
st.markdown("""
<style>
    .stVideo {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 5px;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Başlık
st.title("🎯 YOLO BoTSORT Tracker")
st.markdown("### Gerçek Zamanlı Nesne Takibi")

# Model yükleme
@st.cache_resource
def load_model(model_path='UAV-YOLOv11m.pt'):
    """Model'i yükle ve cache'le"""
    try:
        model = YOLO(model_path)
        return model, None
    except Exception as e:
        return None, str(e)

# Video işleme fonksiyonu
def process_video_frame(model, frame, conf_threshold=0.25):
    """Tek bir frame'i işle"""
    results = model.track(
        source=frame,
        persist=True,
        tracker='botsort.yaml',
        conf=conf_threshold,
        verbose=False,
        stream=True
    )
    
    # Stream modunda ilk sonucu al
    for r in results:
        annotated_frame = r.plot()
        return annotated_frame
    
    return frame

# Yan panel - Ayarlar
with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    # Model seçimi
    model_options = ["UAV-YOLOv11m.pt"]
    selected_model = st.selectbox("Model Seçimi", model_options)
    
    # Güven eşiği
    confidence = st.slider("Güven Eşiği", 0.0, 1.0, 0.25, 0.05)
    
    # Model bilgisi
    st.markdown("---")
    st.markdown("### 📊 Model Bilgisi")
    model, error = load_model(selected_model)
    if error:
        st.error(f"Model yüklenemedi: {error}")
    else:
        st.success(f"✅ {selected_model} yüklendi")

# Ana içerik
tab1, tab2, tab3 = st.tabs(["📹 Kamera", "📁 Video Yükle", "ℹ️ Hakkında"])

# Kamera sekmesi
with tab1:
    st.markdown("### 📹 Canlı Kamera Akışı")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        camera_on = st.checkbox("Kamerayı Başlat", key="camera")
        show_fps = st.checkbox("FPS Göster", value=True)
        camera_index = st.selectbox("Kamera Seçimi", [0, 1, 2], help="Farklı kamera indekslerini deneyin")
        
    if camera_on and model:
        # Kamera placeholder'ı
        camera_placeholder = col1.empty()
        fps_placeholder = col2.empty()
        status_placeholder = col2.empty()
        
        # OpenCV kamera başlat
        status_placeholder.info("Kamera açılıyor...")
        cap = cv2.VideoCapture(camera_index)
        
        # Kamera açıldı mı kontrol et
        if not cap.isOpened():
            status_placeholder.error(f"Kamera {camera_index} açılamadı!")
            # Farklı backend'leri dene
            for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2]:
                cap = cv2.VideoCapture(camera_index, backend)
                if cap.isOpened():
                    status_placeholder.success(f"Kamera {camera_index} açıldı (backend: {backend})")
                    break
        
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            status_placeholder.success("Kamera hazır!")
        
        # FPS hesaplama
        prev_time = time.time()
        fps_list = []
        
        if cap.isOpened():
            frame_error_count = 0
            while camera_on:
                ret, frame = cap.read()
                if not ret or frame is None:
                    frame_error_count += 1
                    if frame_error_count > 10:
                        status_placeholder.error("Kamera görüntüsü alınamadı! Kamera bağlantısını kontrol edin.")
                        break
                    continue
                
                frame_error_count = 0  # Reset error count on successful frame
                
                # Frame boyutunu kontrol et
                if frame.shape[0] == 0 or frame.shape[1] == 0:
                    continue
                
                try:
                    # Frame'i işle
                    processed_frame = process_video_frame(model, frame, confidence)
                except Exception as e:
                    status_placeholder.error(f"Frame işleme hatası: {str(e)}")
                    processed_frame = frame
                
                # BGR'den RGB'ye çevir
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # FPS hesapla
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                fps_list.append(fps)
                if len(fps_list) > 30:
                    fps_list.pop(0)
                avg_fps = sum(fps_list) / len(fps_list)
                
                # FPS'i frame'e ekle
                if show_fps:
                    cv2.putText(processed_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Görüntüyü göster
                camera_placeholder.image(processed_frame, channels="RGB", use_container_width=True)
                
                # FPS bilgisi
                if show_fps:
                    fps_placeholder.metric("FPS", f"{avg_fps:.1f}")
                
                # Streamlit'in yenilenmesi için
                time.sleep(0.001)
                
                # Kamera durumu kontrol
                if not st.session_state.get('camera', False):
                    break
        else:
            status_placeholder.error("Hiçbir kamera bulunamadı! Lütfen:")
            st.markdown("""
            - Kamera bağlantısını kontrol edin
            - Başka uygulamalar kamerayı kullanıyor olabilir
            - Farklı kamera indekslerini deneyin (0, 1, 2)
            - Windows'ta kamera izinlerini kontrol edin
            """)
        
        if 'cap' in locals():
            cap.release()

# Video yükleme sekmesi
with tab2:
    st.markdown("### 📁 Video Dosyası İşle")
    
    uploaded_file = st.file_uploader("Video seçin", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_file is not None and model:
        # Video bilgileri
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"📄 Dosya: {uploaded_file.name}")
        with col2:
            st.info(f"📏 Boyut: {uploaded_file.size / 1024 / 1024:.1f} MB")
        
        # İşleme butonu
        if st.button("🚀 İşlemeyi Başlat", type="primary"):
            # Progress bar ve preview alanları
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("#### 📊 İşleme Durumu")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
            with col_right:
                st.markdown("#### 👁️ Canlı Önizleme")
                preview_placeholder = st.empty()
            
            # Geçici dosya oluştur
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
                tmp_input.write(uploaded_file.read())
                input_path = tmp_input.name
            
            # Çıktı dosyası
            output_path = tempfile.mktemp(suffix='_tracked.mp4')
            
            # Video bilgilerini al
            cap = cv2.VideoCapture(input_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # Video işleme
            status_text.text("Video işleniyor...")
            
            # Video writer için hazırlık
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # YOLO tracking - stream modunda çalıştır
            results = model.track(
                source=input_path,
                save=False,  # Kendimiz kaydedeceğiz
                tracker='botsort.yaml',
                conf=confidence,
                verbose=False,
                stream=True
            )
            
            # Stream sonuçlarını işle
            frame_count = 0
            for r in results:
                frame_count += 1
                
                # Frame'i işle ve kaydet
                annotated_frame = r.plot()
                out_writer.write(annotated_frame)
                
                # Her 10 frame'de bir önizleme göster
                if frame_count % 10 == 0:
                    # BGR'den RGB'ye çevir önizleme için
                    preview_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    # Boyutu küçült hız için
                    preview_frame = cv2.resize(preview_frame, (640, 480))
                    preview_placeholder.image(preview_frame, channels="RGB", use_container_width=True)
                
                # Progress güncelle
                if total_frames > 0:
                    progress = int((frame_count / total_frames) * 100)
                    progress_bar.progress(min(progress, 99))
                    status_text.text(f"İşleniyor... {frame_count}/{total_frames} frame")
            
            # Video writer'ı kapat
            out_writer.release()
            
            # İşleme tamamlandı
            progress_bar.progress(100)
            status_text.success("✅ Video işleme tamamlandı!")
            
            # Sonuç videosunu göster
            st.markdown("### 🎬 İşlenmiş Video")
            
            # Bizim kaydettiğimiz video dosyasını kullan
            if Path(output_path).exists():
                # Video önizleme - video dosyasını oku ve göster
                try:
                    with open(output_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                    st.video(video_bytes)
                
                    # İndirme butonu
                    st.download_button(
                        label="📥 İşlenmiş Videoyu İndir",
                        data=video_bytes,
                        file_name=f"tracked_{uploaded_file.name}",
                        mime="video/mp4"
                    )
                except Exception as e:
                    st.error(f"Video gösteriminde hata: {str(e)}")
                    st.info(f"Video dosyası yolu: {output_path}")
                
                # İstatistikler
                st.markdown("### 📊 İşleme İstatistikleri")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Toplam Frame", total_frames)
                with col2:
                    st.metric("FPS", fps)
                with col3:
                    st.metric("Çözünürlük", f"{width}x{height}")
            
            else:
                st.error("İşlenmiş video dosyası bulunamadı!")
                st.info(f"Beklenen konum: {output_path}")
            
            # Geçici dosyaları temizle (output hariç - kullanıcı indirmek isteyebilir)
            Path(input_path).unlink(missing_ok=True)

# Hakkında sekmesi
with tab3:
    st.markdown("""
    ### 🎯 UAV-YOLOv11m BoTSORT Drone Tracker 
    
    #### 🛠️ Kullanılan Teknolojiler:
    - **UAV-YOLOv11m**: Drone için gelişmiş YOLO modeli
    - **BoTSORT**: Kamera hareketi telafisi ile gelişmiş tracking
    - **Streamlit**: Modern web arayüzü
    - **OpenCV**: Video işleme
    
    #### 📝 Kullanım:
    1. Sol panelden model ve güven eşiği seçin
    2. Kamera sekmesinden canlı takip yapın
    3. Video yükle sekmesinden dosya işleyin
    """)
    
    # BoTSORT parametreleri
    with st.expander("⚙️ BoTSORT Parametreleri"):
        st.code("""
# botsort.yaml içeriği:
tracker_type: botsort
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: 30
match_thresh: 0.8
        """, language='yaml')