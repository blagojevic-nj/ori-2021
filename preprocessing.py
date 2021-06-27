import numpy as np
import PIL.Image
import os
import pickle


def load_data(data, subdir_name, image_size):
    pickle_data_filename = os.path.abspath('.') + '\\' + subdir_name + '.p'
    print(pickle_data_filename)
    if os.path.isfile(pickle_data_filename):
        imgs = pickle.load(open(subdir_name + '.p', "rb"))
    else:
        shape = (len(data), image_size, image_size, 3)
        imgs = np.zeros(shape)
        for i in range(shape[0]):
            filename = os.path.abspath('.') + '\\Data\\' + subdir_name + '\\' + data[i]
            image = PIL.Image.open(filename)
            image = image.resize((image_size, image_size))
            image = np.array(image)
            image = np.clip(image / 255.0, 0.0, 1.0)
            imgs[i] = image

        pickle.dump(imgs, open(subdir_name + '.p', "wb"))

    return imgs


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
