FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit-app.py", "--server.port=8501", "--server.enableCORS=false"]
