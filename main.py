import pandas as pd
import numpy as np
import os

import keras
from keras.layers import Dense

from keras.applications import xception
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l1_l2

from sklearn.model_selection import train_test_split
import preprocessing


def add_dense(base_model):
    x = base_model.output

    outputs = Dense(10, activation='softmax', bias_regularizer=l1_l2())(x)

    mod = keras.models.Model(inputs=base_model.input, outputs=outputs)

    for layer in mod.layers[:40]:
        layer.trainable = False

    return mod


if __name__ == '__main__':
    train_data = os.listdir("Data/train")

    images_filtered = preprocessing.load_data(train_data, "train", 100)
    labels_raw = pd.read_csv("Data/labels.csv", sep=',', header=0, quotechar='"')
    filtered_labels = preprocessing.top_breeds(labels_raw)
    final_list = np.concatenate((filtered_labels[1], filtered_labels[1]))

    clss_bin = preprocessing.classes_binary(final_list)

    num_validation = 0.30
    x_train, x_validation, y_train, y_validation = train_test_split(images_filtered, clss_bin,
                                                                    test_size=num_validation, random_state=3)

    x_test = x_validation[len(x_validation) * 2 // 3:]
    y_test = y_validation[len(y_validation) * 2 // 3:]
    x_validation = x_validation[:len(x_validation) * 2 // 3]
    y_validation = y_validation[:len(y_validation) * 2 // 3]

    model = xception.Xception(weights='imagenet', include_top=False, input_shape=x_train.shape[1:], pooling='max')

    model = add_dense(model)

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1, callbacks=[es_callback], validation_data=(x_validation, y_validation))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
