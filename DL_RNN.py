# Bidirectional LSTM

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# toy dataset: IMDB (binary sentiment)
max_tokens = 20000
max_len = 200

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_tokens)
word_index = keras.datasets.imdb.get_word_index()

# pad/truncate
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test  = keras.preprocessing.sequence.pad_sequences(x_test,  maxlen=max_len)

# model
model = keras.Sequential([
    layers.Embedding(input_dim=max_tokens, output_dim=128, input_length=max_len, mask_zero=True),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy"])

history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=5,
    validation_split=0.2,
    callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)]
)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy:", acc)

