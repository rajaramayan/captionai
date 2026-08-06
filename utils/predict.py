import os
import numpy as np
import tensorflow as tf
from utils.cnn import build_cnn_extractor, extract_features
from utils.tokenizer import load_tokenizer
from utils.preprocess import pad_sequence
import time

# Global references to avoid reloading
_cnn_model = None
_lstm_model = None
_tokenizer = None
MAX_LENGTH = 34  # Standard for Flickr8k

# Always resolve models/ relative to the project root (parent of utils/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_PROJECT_ROOT, 'models')

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

    # Load Tokenizer
    if _tokenizer is None:
        tokenizer_path = os.path.join(_MODELS_DIR, 'tokenizer.pkl')
        _tokenizer = load_tokenizer(tokenizer_path)
        if _tokenizer:
            print(f"Tokenizer loaded with {len(_tokenizer.word_index)} words.")
        else:
            print("Tokenizer not found.")

    # Load LSTM
    if _lstm_model is None and _tokenizer is not None:
        lstm_path = os.path.join(_MODELS_DIR, 'model_lstm.h5')
        if os.path.exists(lstm_path):
            try:
                print("Loading LSTM model...")
                _lstm_model = tf.keras.models.load_model(lstm_path)
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
