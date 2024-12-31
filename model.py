import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape, units=50, dropout_rate=0.2):
    """
    Builds and returns an LSTM model with specified input shape.
    
    :param input_shape: tuple like (sequence_length, 1)
    :param units: number of LSTM units
    :param dropout_rate: dropout fraction
    :return: compiled LSTM model
    """
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    # Second LSTM layer
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(dropout_rate))

    # Final output layer
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model
