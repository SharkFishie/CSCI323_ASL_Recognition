"""Train the ASL hand-skeleton classifier for the live webcam demo.

A small CNN over MediaPipe skeleton images (24 static letters). Skeletons are
clean, high-contrast, low-texture images, so a compact model trains quickly on
CPU and generalizes well. The model and class-name order are saved for
src/live_inference.py.

Prep the data first (scripts/prep_wireframes.py), then run from the repo root:
    .venv311/bin/python -m scripts.train_asl
"""

import json
import os

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.config import (
    BATCH_SIZE,
    CLASS_NAMES_PATH,
    EPOCHS,
    IMAGE_SIZE,
    MODEL_PATH,
    PROCESSED_DIR,
)


def build_model(num_classes):
    """Compact CNN sized for simple, high-contrast skeleton images."""
    model = models.Sequential(
        [
            layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
            # A left/right flip of a hand is the same sign (other hand), so this
            # makes the model robust to the mirrored/selfie webcam orientation.
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.06),
            layers.Rescaling(1.0 / 255),

            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),

            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    if not os.path.isdir(PROCESSED_DIR):
        raise SystemExit(
            f"Processed data not found at '{PROCESSED_DIR}'. "
            "Run `python -m scripts.prep_wireframes` first."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DIR,
        label_mode="categorical",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        subset="training",
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DIR,
        label_mode="categorical",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        subset="validation",
        seed=42,
    )

    class_names = train_ds.class_names
    print(f"{len(class_names)} classes: {class_names}")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(2000).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(CLASS_NAMES_PATH, "w") as fh:
        json.dump(class_names, fh, indent=2)
    print(f"Saved class names -> {CLASS_NAMES_PATH}")

    model = build_model(len(class_names))
    model.summary()

    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    print(f"Done. Best model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
