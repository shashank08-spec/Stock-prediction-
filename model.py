from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
import numpy as np

def build_lstm_model(input_shape, future_days=5):
    """Builds and compiles the LSTM model."""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=future_days) # Output is future_days sequence
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    """Trains the model."""
    history = model.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=0.1,
        verbose=1 # Let standard output catch it
    )
    return model, history

def predict_future(model, scaled_data, sequence_length=60):
    """Predicts the future values based on the last sequence."""
    last_sequence = scaled_data[-sequence_length:]
    last_sequence = np.reshape(last_sequence, (1, sequence_length, 1))
    
    predicted_scaled = model.predict(last_sequence)
    return predicted_scaled
