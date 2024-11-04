import tensorflow as tf
import numpy as np

def create_nn():
    model = tf.keras.Sequential([
        # Flatten the 6x7 board (42 inputs) + 1 input for the current player (total 43 inputs)
        tf.keras.layers.InputLayer(input_shape=(43,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(7, activation='softmax')  # 7 outputs, one for each column
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

def predict_move(model, board, current_player):
    board_flat = board.flatten()  
    input_data = np.append(board_flat, current_player)  

    input_reshaped = input_data.reshape(1, 43)  

    # Get the model's output
    output = model.predict(input_reshaped).flatten()
    return output 