import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    MultiHeadAttention,
    LayerNormalization,
)
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import AdamW
import pickle
import config
import numpy as np


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(
        x, x
    )
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="gelu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res


def build_model(input_shape):
    inputs = Input(shape=input_shape)

    # LSTM Layer
    x = LSTM(128, return_sequences=True, kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(
        inputs
    )
    x = Dropout(0.3)(x)

    # Transformer Encoder
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=256, dropout=0.2)

    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Dense Layers
    x = Dense(64, activation="gelu")(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="gelu")(x)

    # Output
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=output)

    optimizer = AdamW(learning_rate=config.LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.F1Score(name="f1_score"),
        ],
    )

    return model


def train():
    with open("data/processed_data.pkl", "rb") as f:
        X, y = pickle.load(f)

    from sklearn.utils.class_weight import compute_class_weight

    class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    model = build_model((X.shape[1], X.shape[2]))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=20, monitor="val_auc", mode="max", restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "model/best_model.h5", save_best_only=True, monitor="val_auc", mode="max"
        ),
    ]

    history = model.fit(
        X[:-100],
        y[:-100],
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=config.TEST_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
    )

    # Load model terbaik
    model = tf.keras.models.load_model("model/best_model.h5")

    # Evaluasi
    evaluation_results = model.evaluate(X[-100:], y[-100:])

    print("\n=== Evaluation Metrics ===")
    print(f"Validation Loss: {evaluation_results[0]:.4f}")
    print(f"Validation Accuracy: {evaluation_results[1]:.4f}")
    print(f"Validation Precision: {evaluation_results[2]:.4f}")
    print(f"Validation Recall: {evaluation_results[3]:.4f}")
    print(f"Validation AUC: {evaluation_results[4]:.4f}")

    model.save("model/lstm_model.h5")


if __name__ == "__main__":
    train()
