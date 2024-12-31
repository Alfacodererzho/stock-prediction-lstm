import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def predict_and_plot(model_path, scaler, X_test, y_test):
    """
    Loads a saved model, makes predictions on X_test, 
    and plots the results against y_test.
    model_path: path to .h5 file
    scaler: a fitted MinMaxScaler instance
    X_test: test features
    y_test: test labels (scaled)
    """
    # 1. Load model
    model = tf.keras.models.load_model(model_path)

    # 2. Make predictions
    predictions = model.predict(X_test)

    # 3. Inverse transform predictions and true values
    predictions_inv = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # 4. Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_inv, label='Actual Price')
    plt.plot(predictions_inv, label='Predicted Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    return predictions_inv, y_test_inv
