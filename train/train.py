import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from train.dataset import load_set, load_clean_descriptions, load_doc
from utils.tokenizer import save_tokenizer
from utils.lstm import build_lstm_model
from utils.preprocess import generate_sequences, preprocess_caption
from utils.cnn import build_cnn_extractor, extract_features

def data_generator(descriptions, features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key, desc_list in descriptions.items():
            n += 1
            if key not in features:
                continue
            feature = features[key][0]
            for desc in desc_list:
                in_img, in_seq, out_word = generate_sequences(tokenizer, max_length, feature, desc, vocab_size)
                if len(in_img) > 0:
                    X1.extend(in_img)
                    X2.extend(in_seq)
                    y.extend(out_word)
            
            if n == batch_size:
                yield [np.array(X1), np.array(X2)], np.array(y)
                X1, X2, y = list(), list(), list()
                n = 0

def load_captions(filename):
    doc = load_doc(filename)
    mapping = dict()
    # Skip header if it exists
    lines = doc.split('\n')
    if 'image,caption' in lines[0]:
        lines = lines[1:]
        
    for line in lines:
        if len(line) < 2:
            continue
        # Support both comma-separated and space-separated
        if ',' in line:
            tokens = line.split(',', 1)
        else:
            tokens = line.split('\t')
            if len(tokens) < 2:
                tokens = line.split(' ', 1)
                
        if len(tokens) < 2:
            continue
            
        image_id, image_desc = tokens[0], tokens[1]
        image_id = image_id.split('.')[0]
        
        if image_id not in mapping:
            mapping[image_id] = list()
        mapping[image_id].append(preprocess_caption(image_desc))
    return mapping

def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length_func(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

def main():
    print("WARNING: This is the training script.")
    dataset_dir = r'E:\Downloads\Yubraj\flickr8k'
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory '{dataset_dir}' not found.")
        return
        
    print("Initiating Tokenizer build and Model training preparation...")
    
    # 1. Load Descriptions
    captions_file = os.path.join(dataset_dir, 'captions.txt')
    if not os.path.exists(captions_file):
        print(f"Cannot find captions file at {captions_file}")
        return
        
    print("Loading captions...")
    train_descriptions = load_captions(captions_file)
    print(f"Loaded {len(train_descriptions)} images with captions.")
    
    # 2. Tokenizer
    print("Building tokenizer...")
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max_length_func(train_descriptions)
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Max Sequence Length: {max_length}")
    
    # Save Tokenizer
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    save_tokenizer(tokenizer, os.path.join(models_dir, 'tokenizer.pkl'))
    print("Tokenizer saved to models/tokenizer.pkl")
    
    # 3. Extract Features (Simplified for memory)
    # Note: In a real run on 8k images, you would extract all to a .pkl file first. 
    # Here we simulate by just training on a tiny subset to ensure the code executes cleanly.
    print("Extracting CNN features for a small subset of images for demonstration...")
    image_dir = os.path.join(dataset_dir, 'Images')
    cnn_model = build_cnn_extractor()
    features = dict()
    
    subset_keys = list(train_descriptions.keys())[:1000] # TRAIN ON 1000 IMAGES FOR DEMO
    train_descriptions = {k: train_descriptions[k] for k in subset_keys}
    
    import glob
    for img_path in glob.glob(os.path.join(image_dir, '*.jpg')):
        img_id = os.path.basename(img_path).split('.')[0]
        if img_id in subset_keys:
            feat = extract_features(img_path, cnn_model)
            features[img_id] = feat
            
    print(f"Extracted features for {len(features)} images.")

    # 4. Build LSTM Model
    print("Building LSTM model...")
    model = build_lstm_model(vocab_size, max_length)
    
    # 5. Train
    print("Starting training...")
    epochs = 15
    batch_size = 8
    steps = len(train_descriptions) // batch_size
    
    generator = data_generator(train_descriptions, features, tokenizer, max_length, vocab_size, batch_size)
    model.fit(generator, epochs=epochs, steps_per_epoch=steps, verbose=1)
    
    # 6. Save Model
    model_path = os.path.join(models_dir, 'model_lstm.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    print("Training complete! You can now use the Web UI to predict captions.")

if __name__ == '__main__':
    main()
