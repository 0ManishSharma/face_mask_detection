import os
import cv2
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from dataclasses import dataclass

# ---------------- CONFIG ---------------- #

@dataclass
class ModelTrainerConfig:
    img_size: int = 224
    batch_size: int = 16
    epochs: int = 10

    train_images: str = "artifacts/transformed_data/train/images"
    train_annotations: str = "artifacts/transformed_data/train/annotations"

    val_images: str = "artifacts/transformed_data/val/images"
    val_annotations: str = "artifacts/transformed_data/val/annotations"

    model_path: str = "artifacts/model/face_mask_detector.h5"


# ---------------- XML PARSER ---------------- #

class XMLParser:
    class_map = {
        "with_mask": 0,
        "without_mask": 1,
        "mask_weared_incorrect": 2,
    }

    @staticmethod
    def parse(xml_path, img_w, img_h):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        obj = root.find("object")
        label_name = obj.find("name").text
        label = XMLParser.class_map[label_name]

        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text) / img_w
        ymin = int(bbox.find("ymin").text) / img_h
        xmax = int(bbox.find("xmax").text) / img_w
        ymax = int(bbox.find("ymax").text) / img_h

        return label, [xmin, ymin, xmax, ymax]


# ---------------- DATA LOADER ---------------- #

class DatasetLoader:
    def __init__(self, image_dir, ann_dir, img_size):
        self.image_dir = image_dir
        self.ann_dir = ann_dir
        self.img_size = img_size

    def load(self):
        X, y_cls, y_bbox = [], [], []

        for img_name in os.listdir(self.image_dir):
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(self.image_dir, img_name)
            xml_path = os.path.join(
                self.ann_dir, img_name.rsplit(".", 1)[0] + ".xml"
            )

            if not os.path.exists(xml_path):
                continue

            img = cv2.imread(img_path)
            h, w, _ = img.shape
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img / 255.0

            label, bbox = XMLParser.parse(xml_path, w, h)

            X.append(img)
            y_cls.append(label)
            y_bbox.append(bbox)

        print(f"Loaded {len(X)} samples from {self.image_dir}")

        return (
            np.array(X, dtype=np.float32),
            np.array(y_cls, dtype=np.int32),
            np.array(y_bbox, dtype=np.float32),
        )


# ---------------- MODEL ---------------- #

def build_model(img_size):
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))

    x = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)

    class_output = tf.keras.layers.Dense(
        3, activation="softmax", name="class_output"
    )(x)

    bbox_output = tf.keras.layers.Dense(
        4, activation="linear", name="bbox_output"
    )(x)

    model = tf.keras.Model(inputs, [class_output, bbox_output])
    return model


# ---------------- TRAINER ---------------- #

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self):
        print("🚀 Starting Model Training")

        train_loader = DatasetLoader(
            self.config.train_images,
            self.config.train_annotations,
            self.config.img_size,
        )

        val_loader = DatasetLoader(
            self.config.val_images,
            self.config.val_annotations,
            self.config.img_size,
        )

        X_train, y_train_cls, y_train_bbox = train_loader.load()
        X_val, y_val_cls, y_val_bbox = val_loader.load()

        assert len(X_train) > 0, "❌ No training data loaded"
        assert len(X_val) > 0, "❌ No validation data loaded"

        model = build_model(self.config.img_size)

        model.compile(
            optimizer="adam",
            loss={
                "class_output": "sparse_categorical_crossentropy",
                "bbox_output": "mse",
            },
            metrics={"class_output": "accuracy"},
        )

        model.fit(
            X_train,
            {
                "class_output": y_train_cls,
                "bbox_output": y_train_bbox,
            },
            validation_data=(
                X_val,
                {
                    "class_output": y_val_cls,
                    "bbox_output": y_val_bbox,
                },
            ),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
        )

        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
        model.save(self.config.model_path)

        print(f"✅ Model saved at {self.config.model_path}")
