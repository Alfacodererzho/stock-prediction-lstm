from train import train_model
from predict import predict_and_plot

def main():
    # 1. Train model on a specific ticker
    model, scaler, (X_train, X_test, y_train, y_test) = train_model(
        ticker="AAPL",
        start_date="2018-01-01",
        end_date="2023-01-01",
        sequence_length=60,
        epochs=20,
        batch_size=32,
        save_path="lstm_model.h5"
    )

    # 2. Predict on the test set and plot
    predict_and_plot(model_path="lstm_model.h5", 
                     scaler=scaler, 
                     X_test=X_test, 
                     y_test=y_test)

if __name__ == "__main__":
    main()
