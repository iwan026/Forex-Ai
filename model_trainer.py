# model_trainer.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    MultiHeadAttention,
    LayerNormalization,
    TimeDistributed,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.metrics import PrecisionAtRecall
import pickle
import config
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss untuk menangani ketidakseimbangan kelas"""

    def focal_loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon()
        )
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.math.pow(1 - y_pred, gamma)
        return tf.reduce_mean(weight * cross_entropy)

    return focal_loss_fn


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(
        x, x
    )
    x = Dropout(dropout)(x)
    res = x + inputs

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="swish")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res


def build_model(input_shape):
    inputs = Input(shape=input_shape)

    # LSTM Layer
    x = LSTM(
        config.LSTM_UNITS, return_sequences=True, kernel_regularizer=l2(config.L2_REG)
    )(inputs)
    x = Dropout(config.DROPOUT_RATE)(x)

    # Transformer Encoder
    x = transformer_encoder(
        x,
        head_size=64,
        num_heads=config.ATTENTION_HEADS,
        ff_dim=512,
        dropout=config.DROPOUT_RATE,
    )

    # TimeDistributed untuk menjaga informasi temporal
    x = TimeDistributed(Dense(64, activation="swish"))(x)
    x = tf.keras.layers.Flatten()(x)

    # Dense Layers
    x = Dense(128, activation="swish")(x)
    x = Dropout(config.DROPOUT_RATE)(x)
    x = Dense(64, activation="swish")(x)

    # Output
    output = Dense(3, activation="softmax")(x)  # Multi-kelas (buy, sell, neutral)

    model = Model(inputs=inputs, outputs=output)

    optimizer = AdamW(
        learning_rate=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    model.compile(
        optimizer=optimizer,
        loss=focal_loss(),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
            PrecisionAtRecall(recall=0.8, name="prec_at_recall"),
        ],
    )

    return model


def train():
    with open("data/processed_data.pkl", "rb") as f:
        X, y = pickle.load(f)

    # Konversi y ke one-hot untuk multi-kelas
    y = tf.keras.utils.to_categorical(y, num_classes=3)

    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y.argmax(axis=1)), y=y.argmax(axis=1)
    )
    class_weight_dict = {i: class_weights[i] for i in range(3)}

    model = build_model((X.shape[1], X.shape[2]))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=30, monitor="val_auc", mode="max", restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=10, min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            config.BEST_MODEL_PATH, save_best_only=True, monitor="val_auc", mode="max"
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
    model = tf.keras.models.load_model(
        config.BEST_MODEL_PATH, custom_objects={"focal_loss_fn": focal_loss()}
    )

    # Evaluasi
    evaluation_results = model.evaluate(X[-100:], y[-100:])

    print("\n=== Evaluation Metrics ===")
    for metric, value in zip(model.metrics_names, evaluation_results):
        print(f"{metric.capitalize():<20}: {value:.4f}")

    model.save(config.MODEL_PATH, save_format="keras_v3")


if __name__ == "__main__":
    train()
