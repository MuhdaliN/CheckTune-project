# ml/predict.py
import json
import numpy as np
import tensorflow as tf
from ml.utils import extract_mfcc
import argparse

def predict(wav_path, model_path='model_checktune.h5', labels_map='labels_map.json', n_mfcc=40, max_len=174):
    with open(labels_map, 'r') as f:
        mapping = json.load(f)
    classes = mapping['classes']

    mfcc = extract_mfcc(wav_path, n_mfcc=n_mfcc, max_len=max_len)
    x = mfcc[np.newaxis, ..., np.newaxis]  # (1,n_mfcc,max_len,1)
    model = tf.keras.models.load_model(model_path)
    preds = model.predict(x)[0]
    idx = np.argmax(preds)
    return classes[idx], float(preds[idx]), {classes[i]: float(preds[i]) for i in range(len(classes))}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('wav', help='path to wav file')
    parser.add_argument('--model', default='model_checktune.h5')
    parser.add_argument('--labels', default='labels_map.json')
    args = parser.parse_args()
    label, score, all_scores = predict(args.wav, args.model, args.labels)
    print(label, score)
    print(all_scores)
