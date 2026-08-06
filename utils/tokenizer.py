import json
import os


def save_tokenizer(tokenizer, tokenizer_path='models/tokenizer.pkl'):
    """Save tokenizer word_index as JSON to avoid keras.src pickle issues."""
    json_path = tokenizer_path.replace('.pkl', '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer.word_index, f)
    print(f"Tokenizer saved to {json_path}")


def load_tokenizer(tokenizer_path='models/tokenizer.pkl'):
    """Load tokenizer from JSON word_index file."""
    json_path = tokenizer_path.replace('.pkl', '.json')
    if not os.path.exists(json_path):
        print(f"Tokenizer JSON not found at {json_path}")
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        word_index = json.load(f)
    # Create a lightweight wrapper with the same interface as Keras Tokenizer
    return SimpleTokenizer(word_index)


class SimpleTokenizer:
    """Lightweight tokenizer that wraps a word_index dict."""

    def __init__(self, word_index):
        self.word_index = word_index
        self.index_word = {v: k for k, v in word_index.items()}

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = [self.word_index[w] for w in text.split() if w in self.word_index]
            sequences.append(seq)
        return sequences
