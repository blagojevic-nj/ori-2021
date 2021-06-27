import pandas as pd
import os

from sklearn.model_selection import train_test_split
import preprocessing

if __name__ == '__main__':
    train_data = os.listdir("Data/train")
    images = preprocessing.load_data(train_data, "train", 100)

    labels_raw = pd.read_csv("Data/labels.csv", sep=',', header=0, quotechar='"')

    filtered_labels = preprocessing.top_breeds(labels_raw)
    images_filtered = images[filtered_labels[0], :, :, :]
    clss_bin = preprocessing.classes_binary(filtered_labels[1])

    num_validation = 0.30
    x_train, x_validation, y_train, y_validation = train_test_split(images_filtered, clss_bin,
                                                                    test_size=num_validation, random_state=6)
