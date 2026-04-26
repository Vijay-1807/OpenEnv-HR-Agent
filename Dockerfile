# Hugging Face Spaces (Docker + Streamlit) and any Docker Streamlit deploy.
# HF sets PORT; default 7860 matches Space routing.
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-7860} --server.address=0.0.0.0 --server.headless true --browser.gatherUsageStats false --server.fileWatcherType none"]
