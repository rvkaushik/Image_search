import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, MobileNetV3Small, vgg16

def create_dataset_imagegen(train_path):
    """

    :param train_path:
    :return:
    """
    datagen = ImageDataGenerator(rescale=1/255.0, validation_split=.2, horizontal_flip=True, rotation_range=20, preprocessing_function=vgg16.preprocess_input)
    train_gen = datagen.flow_from_directory(train_path, color_mode='rgb', class_mode='categorical', batch_size=32,subset="training")
    valid_gen = datagen.flow_from_directory(train_path, color_mode='rgb', class_mode='categorical', batch_size=32,subset="validation")
    return train_gen, valid_gen


def get_vgg16_base():
    """

    :return:
    """
    model_vgg = VGG16(include_top=False, weights="imagenet", input_shape=(256, 256, 3))
    model_vgg.trainable = False
    return model_vgg


def get_model(model_base):
    """

    :param model_base:
    :return:
    """
    model = tf.keras.Sequential([
        model_base,
        tf.keras.layers.Conv2D(4096, kernel_size=(1, 1), activation="relu"),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Conv2D(4096, kernel_size=(1, 1), activation="relu"),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(11, activation="softmax")
    ])
    return model


def compile_model(model, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"]):
    """

    :param model:
    :param optimizer:
    :param loss:
    :param metrics:
    :return:
    """
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


def train(train_dataset, valid_dataset, epochs=20):
    """

    :param train_dataset:
    :param valid_dataset:
    :param epochs:
    :return:
    """
    history = model.fit(train_dataset, epochs=epochs, validation_data=valid_dataset)
    model.save("model")
    return history


if __name__ == "__main__":

    train_dir = "data/train"
    train_dataset, valid_dataset = create_dataset_imagegen(train_dir)
    model_base = get_vgg16_base()
    model = get_model(model_base)
    model = compile_model(model)
    history = train(train_dataset, valid_dataset)
