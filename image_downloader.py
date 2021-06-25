import os
import numpy as np
import json
import urllib.request

META_INFO = "data/meta/json"
PHOTOS_INFO = "data/photos/photos.txt"


def get_photos_info():
    """
    Function to read photos info text file and return key pair of photo id and download link.
    :return:
    """
    photos_info = np.loadtxt(PHOTOS_INFO, delimiter=',', unpack=True, dtype=str, usecols=(0, 1))
    photos_info_dict = dict(zip(photos_info[0], photos_info[1]))
    return photos_info_dict


def get_meta_info(meta_info_dir):
    """
    Functio to return train json and test jsons
    :param meta_info_dir:
    :return:
    """
    files = os.listdir(meta_info_dir)
    retrieval_info = [os.path.join(meta_info_dir, item) for item in files if item.startswith("ret")]
    test_info = [os.path.join(meta_info_dir, item) for item in files if item.startswith("test")]
    return retrieval_info, test_info


def create_images(jsons_info, image_dir, photos_info_dict, num_images=200):
    """
    Function to download images for all categories from json.
    :param jsons_info:
    :param image_dir:
    :param photos_info_dict:
    :param num_images:
    :return:
    """
    for item in jsons_info:
        json_info = json.load(open(item, "r"))
        category_dir = os.path.join(image_dir, os.path.splitext(os.path.basename(item))[0])
        print("Downloading in -- ", category_dir)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
        count = 0
        i = 0
        while count < num_images:
            photo_id = json_info[i]["photo"]
            link = photos_info_dict[f'{photo_id:09}']
            try:
              urllib.request.urlretrieve(link, f"{category_dir}/{count}.jpg")
              count = count + 1
              i = i + 1
            except:
              i = i + 1
    print("Image - Downloaded")


if __name__ == "__main__":

    train_dir = "data/train"
    test_dir = "data/test"
    photos_info_dict = get_photos_info()
    retrieval_info, test_info = get_meta_info(META_INFO)
    create_images(retrieval_info, train_dir, photos_info_dict)
    create_images(test_info, test_dir, photos_info_dict, num_images=2)

