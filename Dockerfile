FROM python:3.12-slim

# System libraries OpenCV and MediaPipe needed at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgles2 libegl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies 
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# The code the API actually needs
COPY api/ ./api/
COPY ml/ ./ml/
COPY vision/ ./vision/
COPY database/ ./database/
COPY config.py .
COPY data/models/ ./data/models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


