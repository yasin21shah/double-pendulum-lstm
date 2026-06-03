import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# create overlapping windows using fast memory views
def create_windows_vectorized(data, window_size, offset_val):
    windows = sliding_window_view(data, window_shape=(window_size, data.shape[1]))
    X = np.squeeze(windows)[:-offset_val]
    y = data[window_size + offset_val - 1:]
    
    # ensure input and label arrays match in length
    min_len = min(len(X), len(y))
    return X[:min_len], y[:min_len]

# build and compile the lstm model
def build_lstm_model(input_dim, output_dim, window_size=15, lstm_units=50):
    model = Sequential([
        Input(shape=(window_size, input_dim)),
        LSTM(lstm_units, return_sequences=False),
        Dense(output_dim, activation="linear")
    ])
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model