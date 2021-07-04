import numpy as np
import pandas as pd
import PIL.Image
import PIL.ImageOps
import PIL.ImageEnhance
import os
import pickle


def load_data(data, subdir_name, image_size):
    print("Loading data...")
    pickle_data_filename = os.path.abspath('.') + '\\' + subdir_name + '.p'
    if os.path.isfile(pickle_data_filename):
        images_filtered = pickle.load(open(subdir_name + '.p', "rb"))
    else:
        shape = (len(data), image_size, image_size, 3)
        imgs = np.zeros(shape)
        imgs_flip = np.zeros(shape)
        for i in range(shape[0]):
            filename = os.path.abspath('.') + '\\Data\\' + subdir_name + '\\' + data[i]
            image = PIL.Image.open(filename)
            image1 = image.resize((image_size, image_size))
            image1 = np.array(image1)
            image1 = np.clip(image1 / 255.0, 0.0, 1.0)
            imgs[i] = image1

            image2 = PIL.ImageOps.mirror(image)
            image2 = image2.resize((image_size, image_size))
            image2 = np.array(image2)
            image2 = np.clip(image2 / 255.0, 0.0, 1.0)
            imgs_flip[i] = image2

        labels_raw = pd.read_csv("Data/labels.csv", sep=',', header=0, quotechar='"')

        filtered_labels = top_breeds(labels_raw)
        images_filtered = imgs[filtered_labels[0], :, :, :]
        images_flip_filtered = imgs_flip[filtered_labels[0], :, :, :]

        images_filtered = np.concatenate((images_filtered, images_flip_filtered))

        pickle.dump(images_filtered, open(subdir_name + '.p', "wb"))

    return images_filtered


def top_breeds(lbls, size=10):
    labels_fr = lbls.groupby('breed').count()
    labels_fr = labels_fr.sort_values(by='id', ascending=False)

    main_lbls = labels_fr.head(size)

    main_lbls_list = []
    for breed in main_lbls.to_csv().split('\n'):
        main_lbls_list.append(breed.split(',')[0])

    main_lbls_list.remove('')
    main_lbls_list.pop(0)
    lbls_numpy = lbls['breed'].to_numpy()
    lbls_numpy = lbls_numpy.reshape(lbls_numpy.shape[0], 1)
    filtered_lbls = np.where(lbls_numpy == main_lbls_list)

    return filtered_lbls


def classes_binary(cls_indexes):
    ret_val = np.zeros((len(cls_indexes), 10))
    cnt = 0
    for idx in cls_indexes:
        tmp_array = np.zeros(10)
        tmp_array[idx] = 1.

        ret_val[cnt] = tmp_array
        cnt += 1
    return ret_val
