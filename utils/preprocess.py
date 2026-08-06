import string
import numpy as np
import tensorflow as tf


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if len(word) > 1]
    text = [word for word in text if word.isalpha()]
    return ' '.join(text)


def preprocess_caption(caption):
    return 'startseq ' + clean_text(caption) + ' endseq'


def pad_sequence(seq, max_length):
    """Simple numpy-based pre-padding (zeros at the start)."""
    padded = np.zeros(max_length, dtype=np.int32)
    seq = seq[-max_length:]          # truncate if longer
    padded[max_length - len(seq):] = seq
    return padded


def generate_sequences(tokenizer, max_length, feature, caption, vocab_size):
    X1, X2, y = list(), list(), list()
    seq = tokenizer.texts_to_sequences([caption])[0]
    for i in range(1, len(seq)):
        in_seq, out_seq = seq[:i], seq[i]
        in_seq = pad_sequence(in_seq, max_length)
        out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
        X1.append(feature)
        X2.append(in_seq)
        y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)
