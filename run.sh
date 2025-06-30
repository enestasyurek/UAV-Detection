#!/bin/bash
# BoTSORT Tracker Streamlit uygulamasını başlatma scripti

echo "🚀 YOLO BoTSORT Tracker başlatılıyor..."

# Gerekli paketleri kontrol et
if ! command -v streamlit &> /dev/null; then
    echo "⚠️ Streamlit bulunamadı. Kurulum için:"
    echo "pip install -r requirements.txt"
    exit 1
fi

# Model dosyasını kontrol et
if [ ! -f "best.pt" ]; then
    echo "⚠️ best.pt model dosyası bulunamadı!"
    echo "Model dosyanızı bu klasöre kopyalayın veya app.py içinde model yolunu güncelleyin."
fi

# Streamlit uygulamasını başlat
streamlit run app.py \
    --server.port 8501 \
    --server.address localhost \
    --browser.serverAddress localhost \
    --server.maxUploadSize 200