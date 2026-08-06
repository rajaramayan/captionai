# An Extension for Image Captioning Using AI Technique

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0.0-black?logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange?logo=tensorflow)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5-purple?logo=bootstrap)

A complete, production-ready AI-based web application that automatically generates meaningful captions for uploaded images using a Deep Learning architecture (CNN-LSTM).

## 🌟 Features

* **Advanced AI Architecture**: Utilizes InceptionV3 (CNN) for feature extraction and Bidirectional LSTM for sequence generation.
* **Modern Web Interface**: Responsive UI built with Bootstrap 5, featuring drag-and-drop uploads, dark mode toggle, and micro-animations.
* **Accessibility**: Integrated Text-to-Speech (TTS) functionality allowing users to listen to generated captions.
* **User Management**: Secure registration, login, and session management using Flask-Login and password hashing.
* **Caption History & Dashboard**: View statistical usage, recent activity, and historical predictions.
* **Export Functionality**: Export your prediction history to CSV format.
* **REST API**: Endpoints for programmatic access to the captioning service.
* **Docker Ready**: Fully containerized environment for seamless deployment to production.
* **Comprehensive Documentation**: See [diagrams.md](diagrams.md) for full System, Flowchart, UML, and ER diagrams.

---

## 🏗 System Architecture

The application implements a standard encoder-decoder workflow:
1. **Image Preprocessing**: Images are resized to 299×299 and normalized.
2. **Encoder (CNN)**: InceptionV3 extracts a 2048-dimensional feature vector.
3. **Decoder (LSTM)**: An Embedding layer followed by a Bidirectional LSTM generates the caption iteratively, word by word, until an `<endseq>` token is reached.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.12
- Git

### Local Setup
1. **Clone the repository** (or download the source):
   ```bash
   cd ImageCaptioning
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize Database and Start Server**:
   ```bash
   python app.py
   ```
   The application will be accessible at `http://localhost:5000`.

---

## 🧠 Model Training (Flickr8k Dataset)

To train the model from scratch, you will need the Flickr8k dataset.

1. **Download Dataset**: Download the `Flickr8k_Dataset` (images) and `Flickr8k_text` (descriptions).
2. **Directory Structure**: Place them in a folder accessible to the project (e.g., inside the `train/` directory).
3. **Run Training Script**:
   ```bash
   python -m train.train
   ```
   *Note: Ensure you update the directory paths inside `train/train.py` to point to your Flickr8k text and images folders.*
4. **Export Models**: Once trained, ensure your `model_lstm.h5` and `tokenizer.pkl` are copied into the `models/` directory of the root project folder.

---

## 🐳 Docker Deployment

The application is bundled with a `Dockerfile` and `gunicorn.conf.py` for immediate deployment.

1. **Build the Image**:
   ```bash
   docker build -t image-captioning .
   ```
2. **Run the Container**:
   ```bash
   docker run -d -p 5000:5000 --name caption-app image-captioning
   ```
   Access the app at `http://localhost:5000`.

### Cloud Deployment (Render/AWS/Railway)
- Connect your GitHub repository to your host.
- Specify the start command as: `gunicorn -c gunicorn.conf.py app:app`
- Ensure you provision sufficient RAM (at least 2GB) since the TensorFlow backend requires memory to load the InceptionV3 model.

---

## 📄 API Documentation

### `POST /upload`
Uploads an image and returns the AI-generated caption.
- **Request Body**: `multipart/form-data` with key `image` containing the image file.
- **Response** (JSON):
  ```json
  {
      "filename": "example.jpg",
      "caption": "a dog running through the grass",
      "confidence": 0.95,
      "time": 1.24
  }
  ```

### `GET /export/csv`
Requires authentication. Downloads a CSV file containing the user's prediction history.

---

## 🔐 Security Information

- Uses `Werkzeug` PBKDF2 for password hashing.
- Secures file uploads by checking `ALLOWED_EXTENSIONS` (`png`, `jpg`, `jpeg`, `webp`) and using `secure_filename`.
- Prevents unauthenticated access via `@login_required` middleware.
