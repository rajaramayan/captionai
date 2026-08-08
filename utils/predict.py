import os
import numpy as np
import tensorflow as tf
from utils.cnn import build_cnn_extractor, extract_features
from utils.tokenizer import load_tokenizer
from utils.lstm import build_lstm_model
from utils.preprocess import pad_sequence
import time
import urllib.request

# Global references to avoid reloading
_cnn_model = None
_lstm_model = None
_tokenizer = None
MAX_LENGTH = 34  # Standard for Flickr8k

# Always resolve models/ relative to the project root (parent of utils/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_PROJECT_ROOT, 'models')


def _download_if_missing(filename, env_var):
    """Download a model file from a URL if it doesn't exist locally.
    
    Set the following environment variables on your cloud platform:
      MODEL_LSTM_URL   -> public URL to model_lstm.h5
      TOKENIZER_URL    -> public URL to tokenizer.json
    """
    path = os.path.join(_MODELS_DIR, filename)
    if not os.path.exists(path):
        url = os.environ.get(env_var)
        if url:
            print(f"Downloading {filename} from {env_var}...")
            os.makedirs(_MODELS_DIR, exist_ok=True)
            urllib.request.urlretrieve(url, path)
            print(f"{filename} downloaded successfully.")
        else:
            print(f"WARNING: {filename} not found locally and {env_var} env var not set.")
    return path

def load_models():
    global _cnn_model, _lstm_model, _tokenizer

    # Load CNN
    if _cnn_model is None:
        try:
            print("Loading InceptionV3 CNN extractor...")
            _cnn_model = build_cnn_extractor()
            print("CNN loaded.")
        except Exception as e:
            print("Error loading CNN model:", e)

    # Load Tokenizer (auto-download from TOKENIZER_URL env var if missing)
    if _tokenizer is None:
        tokenizer_path = _download_if_missing('tokenizer.json', 'TOKENIZER_URL')
        tokenizer_path = tokenizer_path.replace('.json', '.pkl')  # load_tokenizer handles .pkl -> .json
        _tokenizer = load_tokenizer(tokenizer_path)
        if _tokenizer:
            print(f"Tokenizer loaded with {len(_tokenizer.word_index)} words.")
        else:
            print("Tokenizer not found.")

    # Load LSTM (auto-download from MODEL_LSTM_URL env var if missing)
    if _lstm_model is None and _tokenizer is not None:
        lstm_path = _download_if_missing('model_lstm.h5', 'MODEL_LSTM_URL')
        if os.path.exists(lstm_path):
            try:
                print("Loading LSTM model...")
                try:
                    _lstm_model = tf.keras.models.load_model(lstm_path, compile=False)
                except Exception:
                    # Fallback: rebuild architecture and load weights only
                    # (needed when Keras serializes internal ops like NotEqual from mask_zero)
                    vocab_size = len(_tokenizer.word_index) + 1
                    _lstm_model = build_lstm_model(vocab_size, MAX_LENGTH)
                    _lstm_model.load_weights(lstm_path)
                print("LSTM loaded.")
            except Exception as e:
                print("Error loading LSTM model:", e)
        else:
            print(f"LSTM model not found at {lstm_path}")

def index_to_word(integer, tokenizer):
    return tokenizer.index_word.get(integer, None)

def generate_caption(image_path):
    load_models()
    
    # Fallback to dummy if models aren't present (to prevent server crash on fresh clone)
    if _cnn_model is None or _lstm_model is None or _tokenizer is None:
        time.sleep(1.5)
        return "Model not fully loaded. Please place tokenizer.pkl and model_lstm.h5 in the models/ folder.", 0.0

    try:
        # Extract features
        feature = extract_features(image_path, _cnn_model)
        
        in_text = 'startseq'
        for i in range(MAX_LENGTH):
            sequence = _tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequence(sequence, MAX_LENGTH)
            sequence = np.expand_dims(sequence, axis=0)
            
            yhat = _lstm_model.predict([feature, sequence], verbose=0)
            
            # Get highest probability index
            yhat_idx = np.argmax(yhat)
            word = index_to_word(yhat_idx, _tokenizer)
            
            if word is None:
                break
                
            in_text += ' ' + word
            
            if word == 'endseq':
                break
                
        # Clean final caption
        final_caption = in_text.replace('startseq', '').replace('endseq', '').strip()
        
        # Calculate dummy confidence for now, can implement proper softmax probability averaging
        confidence = float(np.max(yhat))
        
        return final_caption, confidence
    except Exception as e:
        print("Error during prediction:", e)
        return "Error predicting caption.", 0.0
