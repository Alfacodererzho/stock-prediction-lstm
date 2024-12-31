import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

from data_preprocessing import load_data, create_sequences
from model import build_lstm_model

def train_model(
    ticker="AAPL", 
    start_date="2018-01-01", 
    end_date="2023-01-01", 
    sequence_length=60,
    epochs=20,
    batch_size=32,
    save_path="lstm_model.h5"
):
    """
    Loads data, creates sequences, builds/trains LSTM, saves the model, 
    and returns history plus train/test sets for further evaluation.
    """

    # 1. Load data
    df = load_data(ticker, start_date, end_date)
    
    # 2. Create sequences
    X, y, scaler = create_sequences(df, sequence_length)

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        shuffle=False  # IMPORTANT for time series; no random shuffling
    )

    # 4. Build the model
    model = build_lstm_model(input_shape=(sequence_length, 1))

    # 5. Train
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, 
        y_train, 
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    
    # 6. Save the model
    model.save(save_path)
    print(f"Model saved to {save_path}")

    # 7. Visualize training process (optional)
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.show()

    return model, scaler, (X_train, X_test, y_train, y_test)
