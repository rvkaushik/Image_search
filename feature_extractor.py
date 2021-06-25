import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import vgg16
from PIL import Image


def get_model(path="model/jio_v1"):
    """

    :param path:
    :return:
    """
    model = tf.keras.models.load_model(path)
    model_1 = tf.keras.Sequential([layer for layer in model.layers[:-3]])
    return model_1


def feature_extractor(model, image_path):
    """

    :param model:
    :param image_path:
    :return:
    """
    img = Image.open(image_path).resize((256, 256))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    features = model.predict(img)
    features = features.flatten()
    features = features/np.linalg.norm(features)
    return features


def get_image_list(path="data/train"):
    """

    :param path:
    :return:
    """
    image_list = []
    for root, sub, files in os.walk(path):
        for item in files:
            if item.endswith(".jpg"):
                image_list.append(os.path.join(root, item))
    return image_list


def create_features(image_list, model, feature_dir="features"):
    """

    :param image_list:
    :return:
    """
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    for img_path in image_list:
        features = feature_extractor(model, img_path)
        feature_path = os.path.join(feature_dir, img_path.split("train/")[-1].replace("/", "_").replace("jpg", "npy"))
        np.save(feature_path, features)
    print("feature created")


if __name__ == "__main__":
    model = get_model()
    image_list = get_image_list()
    create_features(image_list, model)


