# import libraries
import tensorflow as tf
import tensorflow_hub as hub

import argparse
import json
import numpy as np

from PIL import Image

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()

parser.add_argument('--input', default='./test_images/hard-leaved_pocket_orchid.jpg', type = str, action="store", help='Input image path to predict')
parser.add_argument('--model', default='./image_classifier_model.h5', type = str, action="store", help='Saved model path')
parser.add_argument('--top_k', default=5, type=int, action="store", help='Top K most likely classes, default 5')
parser.add_argument('--category_names', default='./label_map.json', type=str, action="store", help='Input path to label flower category names')

arg_parser = parser.parse_args()

image_path = arg_parser.input
model_path = arg_parser.model
top_k = arg_parser.top_k if arg_parser.top_k > 0 else 5
category_names = arg_parser.category_names

def load_saved_model():
    return tf.keras.models.load_model(model_path, custom_objects = {'KerasLayer':hub.KerasLayer}, compile=False)


def load_image_classnames():
    with open(category_names, 'r') as f:
        class_names = json.load(f)
        return class_names


def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    return image.numpy()


def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)
    processed_image = process_image(image)
    expanded_dims_image = np.expand_dims(processed_image, axis=0)
    
    predictions = model.predict(expanded_dims_image)
    probs, labels = tf.nn.top_k(predictions, k=top_k)
    probs = list(probs.numpy()[0])
    labels = list(labels.numpy()[0])
    
    return probs, labels

if __name__== "__main__":
    
    class_names = load_image_classnames()
    loaded_model = load_saved_model()
    
    probs, labels = predict(image_path, loaded_model, top_k)

    print ("\n=======> Top {} class probability <=======\n".format(top_k))

    for i, prob, label in zip(range(top_k), probs, labels):
        print('=========================================\n')
        print('** Prediction: {}'.format(i + 1))
        print('- Image label:', label)
        print('- Image classname:', class_names[str(label+1)].title())
        print('- Image class probability:', prob)
        
    print('=========================================')
    print('\nApplication shutting down...')
        