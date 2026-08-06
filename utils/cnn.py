import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from PIL import Image
import numpy as np

def build_cnn_extractor():
    base_model = InceptionV3(weights='imagenet')
    model = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    return model

def extract_features(img_path, model):
    # Use PIL directly to avoid keras.src.preprocessing issues
    img = Image.open(img_path).convert('RGB')
    img = img.resize((299, 299))
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x, verbose=0)
    return feature
