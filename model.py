import pandas as pd
import numpy as np
import os

from keras.applications import xception
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l1_l2

from sklearn.model_selection import train_test_split
import preprocessing
import matplotlib.pyplot as plt


def add_dense(base_model):
    x = base_model.output
    x = Dense(10, activation='softmax')(x)
    mod = keras.models.Model(inputs=base_model.input, outputs=x)

    for layer in mod.layers[:40]:
        layer.trainable = False

    return mod


def my_model(input_shape):
    my_mod = Sequential()
    my_mod.add(Conv2D(32, (3, 3), input_shape=input_shape))
    my_mod.add(Activation('relu'))
    my_mod.add(MaxPooling2D(pool_size=(2, 2)))

    my_mod.add(Conv2D(64, (3, 3)))
    my_mod.add(Activation('relu'))
    my_mod.add(MaxPooling2D(pool_size=(2, 2)))

    my_mod.add(Conv2D(128, (3, 3)))
    my_mod.add(Activation('relu'))
    my_mod.add(MaxPooling2D(pool_size=(2, 2)))

    my_mod.add(Dropout(rate=0.8))

    my_mod.add(Flatten())
    my_mod.add(Dense(700, bias_regularizer=l1_l2()))
    my_mod.add(Activation('relu'))
    my_mod.add(Dropout(0.5))
    my_mod.add(Dense(10, bias_regularizer=l1_l2()))
    my_mod.add(Activation('sigmoid'))

    return my_mod


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

    # model = my_model(x_train.shape[1:])

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    history = model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=1, callbacks=[es_callback],
                        validation_data=(x_validation, y_validation))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    if score[1] > 0.7:
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)

        model.save("model.h5")
        print("Saved model to disk")
