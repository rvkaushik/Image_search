import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import argparse

from feature_extractor import feature_extractor, get_image_list, get_model

def get_features_list(path="features/"):
    """

    :return:
    """
    features_list = [os.path.join(path, item) for item in os.listdir(path)]
    return features_list

def get_features(features_list, image_dir="data/train"):
    """

    :param features_list:
    :return:
    """
    features = []
    image_list = []
    for f_path in features_list:
        feature = np.load(f_path)
        features.append(feature)
        image_list.append(os.path.join(image_dir, f_path.split("features/")[-1].replace("_", "/").replace("npy", "jpg")))
    features = np.array(features)
    return features, image_list


def get_scores(img_path, image_list, model, features):
    feature = feature_extractor(model, img_path)
    dists = np.linalg.norm(features - feature, axis=1)
    ids = np.argsort(dists)[:5]
    score = [(dists[id], image_list[id]) for id in ids]
    #print(score)
    return score


def compare_results(test_image, scores):
    """

    :param test_image:
    :param scores:
    :return:
    """
    fig = plt.figure(figsize=(10, 7))
    rows = 2
    columns = 3
    fig.add_subplot(rows, columns, 1)
    img = Image.open(test_image).resize((256,256))
    plt.imshow(img)
    plt.axis('off')
    plt.title("test image")
    for i, (score, img_path) in enumerate(scores[1:]):
        fig.add_subplot(rows, columns, i+2)
        img = Image.open(img_path).resize((256, 256))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Results - {i}")
    out_path = test_image.replace(".jpg", "_output.jpg")
    print(out_path)
    plt.savefig(out_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-ti", "--TestImageDir", help="Provide test Images Directory")
    args = parser.parse_args()

    model = get_model()
    features_list = get_features_list()
    features, image_list = get_features(features_list)
    # image_list = get_image_list()
    test_image_dir = args.TestImageDir

    test_images = os.listdir(test_image_dir)
    for test_image in test_images:
        test_image = os.path.join(test_image_dir, test_image)
        if test_image.endswith("jpg"):
            scores = get_scores(test_image, image_list, model, features)
            compare_results(test_image, scores)
        else:
            print(f"Skipping -- {test_image}, Module only test for jpg images")
    print("Done")
