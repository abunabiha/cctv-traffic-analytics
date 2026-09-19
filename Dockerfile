# ==============================================================================
# Dockerfile: Smart CCTV Analytics & Anomaly Detection System
# Multi-kamera Traffic Analytics (Sudirman ID 282, MBK ID 190, Unila ID 312)
# ==============================================================================

FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install sistem dependensi (FFmpeg untuk HLS, pustaka grafis OpenCV & curl untuk healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Tetapkan direktori kerja
WORKDIR /app

# Salin berkas dependencies terlebih dahulu untuk memanfaatkan Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install PyTorch CPU (ringan ~180MB, cepat & efisien tanpa bloat 3.5GB CUDA GPU drivers)
# kemudian install dependensi aplikasi lainnya
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode program dan aset aplikasi ke dalam container
COPY . /app/

# Pastikan folder logs dan scratch tersedia untuk runtime telemetri
RUN mkdir -p /app/logs /app/scratch /app/scratch/cctv_snaps

# Expose port default Streamlit
EXPOSE 8501

# Healthcheck untuk memonitor ketersediaan aplikasi secara berkala
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Perintah eksekusi utama aplikasi Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
