from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model

def build_lstm_model(vocab_size, max_length):
    # Feature Extractor Model (CNN output is 2048)
    inputs1 = Input(shape=(2048,), name="image_features")
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    # Sequence Model (Text)
    inputs2 = Input(shape=(max_length,), name="text_sequence")
    se1 = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    # Decoder Model (Combining both)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax', name="caption_output")(decoder2)
    
    # Compile Model
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model
