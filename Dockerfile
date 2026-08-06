FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download InceptionV3 weights to prevent timeouts on the first request
RUN python -c "from tensorflow.keras.applications import InceptionV3; InceptionV3(weights='imagenet', include_top=False)"

COPY . .
EXPOSE 5000
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
