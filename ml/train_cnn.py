# ml/train_cnn.py
import os
import numpy as np
from glob import glob
from ml.utils import extract_mfcc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
import argparse
import json

def load_dataset(data_dir, labels, n_mfcc=40, max_len=174):
    X, Y = [], []
    for label in labels:
        pattern = os.path.join(data_dir, label, '*.wav')
        files = glob(pattern)
        for f in files:
            try:
                mfcc = extract_mfcc(f, n_mfcc=n_mfcc, max_len=max_len)
                X.append(mfcc)
                Y.append(label)
            except Exception as e:
                print(f"Failed {f}: {e}")
    X = np.array(X)
    # reshape for conv2d: (samples, n_mfcc, max_len, 1)
    X = X[..., np.newaxis]
    return X, np.array(Y)

def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model

def main(args):
    labels = args.labels.split(',')
    print("Labels:", labels)
    X, Y = load_dataset(args.data_dir, labels, n_mfcc=args.n_mfcc, max_len=args.max_len)
    lb = LabelEncoder()
    y_enc = lb.fit_transform(Y)
    X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    print("Shapes:", X_train.shape, X_val.shape)

    model = build_model(input_shape=X_train.shape[1:], num_classes=len(labels))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.save_model, save_best_only=True, monitor='val_accuracy', mode='max'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
    ]
    history = model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
                        validation_data=(X_val, y_val), callbacks=callbacks)

    # save label encoder mapping
    with open(args.labels_map, 'w') as f:
        json.dump({'classes': list(lb.classes_)}, f)
    print("Training complete. Model saved to", args.save_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/train', help='dataset root with subfolders per label')
    parser.add_argument('--labels', default='cello,clarinet,flute,piano,drums,guitar,saxophone,trumpet,violin')
    parser.add_argument('--n_mfcc', type=int, default=40)
    parser.add_argument('--max_len', type=int, default=174)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_model', default='model_checktune.h5')
    parser.add_argument('--labels_map', default='labels_map.json')
    args = parser.parse_args()
    main(args)

