import os
import sys
import glob
import numpy as np
from nltk.translate.bleu_score import corpus_bleu

sys.path.insert(0, os.getcwd())

from train.train import load_captions, create_tokenizer, max_length_func
from utils.cnn import build_cnn_extractor, extract_features
from utils.preprocess import pad_sequence
from utils.tokenizer import load_tokenizer
from utils.lstm import build_lstm_model
import tensorflow as tf


def load_lstm(models_dir, vocab_size, max_length):
    lstm_path = os.path.join(models_dir, 'model_lstm.h5')
    try:
        model = tf.keras.models.load_model(lstm_path, compile=False)
    except Exception:
        model = build_lstm_model(vocab_size, max_length)
        model.load_weights(lstm_path)
    return model


def generate_caption(feature, tokenizer, lstm_model, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequence(seq, max_length)
        seq = np.expand_dims(seq, axis=0)
        yhat = lstm_model.predict([feature, seq], verbose=0)
        idx = np.argmax(yhat)
        word = tokenizer.index_word.get(idx)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text.replace('startseq', '').strip()


def main():
    dataset_dir   = r'E:\Downloads\Yubraj\flickr8k'
    models_dir    = os.path.join(os.getcwd(), 'models')
    image_dir     = os.path.join(dataset_dir, 'Images')
    captions_file = os.path.join(dataset_dir, 'captions.txt')

    print("Loading captions...")
    descriptions = load_captions(captions_file)
    tokenizer    = load_tokenizer(os.path.join(models_dir, 'tokenizer.pkl'))
    vocab_size   = len(tokenizer.word_index) + 1
    max_length   = max_length_func(descriptions)

    print(f"Vocab size: {vocab_size} | Max length: {max_length}")
    print("Loading models...")
    cnn_model  = build_cnn_extractor()
    lstm_model = load_lstm(models_dir, vocab_size, max_length)

    keys      = list(descriptions.keys())
    # Use images NOT in the training set (train used first 7000)
    train_keys = set(keys[:7000])
    test_keys  = [k for k in keys if k not in train_keys]
    print(f"Test set size: {len(test_keys)} images (held-out from training)")

    actual, predicted = [], []
    processed = 0

    print(f"Evaluating on {len(test_keys)} images...")
    for img_id in test_keys:
        img_path = os.path.join(image_dir, img_id + '.jpg')
        if not os.path.exists(img_path):
            continue
        feature  = extract_features(img_path, cnn_model)
        caption  = generate_caption(feature, tokenizer, lstm_model, max_length)
        refs     = [desc.replace('startseq', '').replace('endseq', '').split()
                    for desc in descriptions[img_id]]
        actual.append(refs)
        predicted.append(caption.split())
        processed += 1
        if processed % 100 == 0:
            print(f"  {processed}/{len(test_keys)} done...")

    print(f"\nEvaluated {processed} images\n")
    print("=" * 40)
    print("        BLEU Score Results")
    print("=" * 40)
    b1 = corpus_bleu(actual, predicted, weights=(1, 0, 0, 0))
    b2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    b3 = corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0))
    b4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    print(f"  BLEU-1 : {b1:.4f}  ({b1*100:.2f}%)")
    print(f"  BLEU-2 : {b2:.4f}  ({b2*100:.2f}%)")
    print(f"  BLEU-3 : {b3:.4f}  ({b3*100:.2f}%)")
    print(f"  BLEU-4 : {b4:.4f}  ({b4*100:.2f}%)")
    print("=" * 40)
    print("\nSample Predictions:")
    print("-" * 40)
    for i in range(min(5, len(predicted))):
        print(f"Generated : {' '.join(predicted[i])}")
        print(f"Reference : {' '.join(actual[i][0])}")
        print()


if __name__ == '__main__':
    main()
