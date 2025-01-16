import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, LSTM, GRU, Dense, TimeDistributed, RepeatVector, Multiply, Permute, Flatten, Dropout, Bidirectional
from keras.optimizers import Adam

def build_lstm_model(input_shape):
    input_layer = Input(shape=input_shape)
    lstm_out = LSTM(64, return_sequences=False)(input_layer)  # Return sequences=False for final output
    dense_out = Dense(64, activation='relu')(lstm_out)
    output_layer = Dense(1)(dense_out)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate= 0.0003), loss='mse', metrics=['mae'])
    return model

def build_bilstm_model(input_shape):
    input_layer = Input(shape=input_shape)
    lstm_out = Bidirectional(LSTM(32, return_sequences=False))(input_layer)  # Return sequences=False for final output
    dense_out = Dense(64, activation='relu')(lstm_out)
    output_layer = Dense(1)(dense_out)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_gru_model(input_shape):
    input_layer = Input(shape=input_shape)
    lstm_out = GRU(32, return_sequences=False)(input_layer)  # Return sequences=False for final output
    dense_out = Dense(64, activation='relu')(lstm_out)
    output_layer = Dense(1)(dense_out)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_bigru_model(input_shape):
    input_layer = Input(shape=input_shape)
    lstm_out = Bidirectional(GRU(64, return_sequences=False))(input_layer)  # Return sequences=False for final output
    dense_out = Dense(64, activation='relu')(lstm_out)
    output_layer = Dense(1)(dense_out)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2), 
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2), 
        Flatten(),
        
        Dense(64, activation='relu'),
        Dropout(0.3),  # Dropout for regularization
        Dense(1, activation='linear')    
        ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_cnn_lstm_model(input_shape):
    input_layer = Input(shape=input_shape)
    cnn1d_out = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
    maxpool_out = MaxPooling1D(pool_size=2)(cnn1d_out)
    lstm_out = LSTM(64, return_sequences=False)(maxpool_out)  # Return sequences=False for final output
    dense_out = Dense(64, activation='relu')(lstm_out)
    output_layer = Dense(1)(dense_out)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_cnn_gru_model(input_shape):
    input_layer = Input(shape=input_shape)
    cnn1d_out = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
    maxpool_out = MaxPooling1D(pool_size=2)(cnn1d_out)
    lstm_out = GRU(64, return_sequences=False)(maxpool_out)  # Return sequences=False for final output
    dense_out = Dense(64, activation='relu')(lstm_out)
    output_layer = Dense(1)(dense_out)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def feature_attention_layer(inputs):
    # Apply Dense layer to compute attention scores for features
    attention = Dense(inputs.shape[-1], activation='tanh')(inputs)  # Shape: (batch, timesteps, features)
    attention = Dense(inputs.shape[-1], activation='softmax')(attention)  # Shape: (batch, timesteps, features)
    attention_output = Multiply()([inputs, attention])  # Element-wise multiplication of features
    return attention_output


def build_lstm_attention_model(input_shape):
    input_layer = Input(shape=input_shape)
    feature_attention_out = feature_attention_layer(input_layer)  # Shape: (batch, timesteps, lstm_units)
    lstm_out = LSTM(64, return_sequences=False, activation='tanh')(feature_attention_out)  # Shape: (batch, timesteps, lstm_units)
    flatten_out = Dense(64, activation='relu')(lstm_out)
    output_layer = Dense(1)(flatten_out)  # Final output layer

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.0003), loss='mse', metrics=['mae'])
    return model


def build_gru_attention_model(input_shape):
    input_layer = Input(shape=input_shape)
    feature_attention_out = feature_attention_layer(input_layer)  # Shape: (batch, timesteps, lstm_units)
    lstm_out = GRU(64, return_sequences=True)(feature_attention_out)  # Shape: (batch, timesteps, lstm_units)
    dense_out = TimeDistributed(Dense(1))(lstm_out)  # Shape: (batch, timesteps, 1)
    flatten_out = Flatten()(dense_out)  # Flatten to (batch, timesteps)
    output_layer = Dense(1)(flatten_out)  # Final output layer

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model



