#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    Function that pre-processes the data model
    Arg:
        X - numpy.ndarray (m, 32, 32, 3) CIFAR 10 data.
        Y - numpy.ndarray (m,) CIFAR 10 labels for X
    Returns:
        X_p, Y_p preproceded
    """

    X_p = K.applications.vgg16.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, num_classes=10)
    return (X_p, Y_p)


def load_dataset():
    """
    Function that loads a CIFAR 10 dataset and generate the Xs Ys sets
    Args:
        None
    Returns:
        A dataset divided in training and test sets
    """

    # Import the CIFAR-10 dataset, normalize data by dividing it to 255
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    # convert from integers to floats and normalizing to range to range 0-1
    X_train = X_train.astype('float16')
    X_test = X_test.astype('float16')
    X_train /= 255
    X_test /= 255

    # one hot encoding target values
    Y_train = K.utils.to_categorical(Y_train, 10)
    Y_test = K.utils.to_categorical(Y_test, 10)

    return X_train, Y_train, X_test, Y_test


def create_my_cnn(Y_train):
    """
    Function that generates the CNN model to work with CIFAR 10 dataset
    Args:
        Y_train
    Returns:
        A partially no trained new model based on VGG16 architecture
    """

    my_vgg = K.applications.vgg16.VGG16(include_top=False,
                                        weights='imagenet',
                                        input_shape=(32, 32, 3),
                                        classes=Y_train.shape[1])

    # Freezing most of the input layers
    for lyr in my_vgg.layers:
        if (lyr.name[0:5] != 'block'):
            lyr.trainable = False

    new_model = K.Sequential()
    new_model.add(my_vgg)
    new_model.add(K.layers.Flatten())
    new_model.add(K.layers.Dense(256,
                                 activation='relu',
                                 kernel_initializer='he_uniform'))
    new_model.add(K.layers.Dense(10, activation='softmax'))
    new_model.summary()

    return new_model


def compile_cnn(a_cnn):
    """
    Function that compiles a NN
    Args:
        a_cnn - New cnn model to be compiled
    Return:
        The compiled cnn
    """

    opt = K.optimizers.SGD(lr=0.001, momentum=0.9)
    a_cnn.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return a_cnn


def train_cnn(a_cnn, X_train, Y_train, X_test, Y_test, bat_, epo_):
    """
    Function that trains a DNN
    Args:
        a_cnn - New cnn model to be compiled
        X_train - x train
        Y_train - y train
        X_test - x test
        Y_test - y test
        batch_size - batch size
        epochs - epochs
    Return:
        The logs of the training
    """

    """
    variable_lr= K.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                               factor=.01,
                                               patience=3,
                                               min_lr=1e-5,
                                               save_best_only=True)

    return a_cnn.fit(X_train,
                     Y_train,
                     epochs=epo_,
                     batch_size=bat_,
                     validation_data=(X_test, Y_test),
                     callbacks=[variable_lr],
                     verbose=1)

    """
    # alternative for data augmentation

    datagen = K.preprocessing.image.ImageDataGenerator(rotation_range=15,
                                                       width_shift_range=0.1,
                                                       height_shift_range=0.1,
                                                       horizontal_flip=True)
    datagen.fit(X_train)
    return a_cnn.fit_generator(datagen.flow(X_train, Y_train, batch_size=bat_),
                               steps_per_epoch=X_train.shape[0] // bat_,
                               epochs=epo_,
                               verbose=1,
                               validation_data=(X_test, Y_test))


if __name__ != '__main__':

    bat_ = 50  # batch size
    epo_ = 50  # epochs

    # load dataset
    X_train, Y_train, X_test, Y_test = load_dataset()

    # define model
    my_cnn = create_my_cnn(Y_train)

    # compile the model
    my_cnn = compile_cnn(my_cnn)

    # Train model
    history = train_cnn(my_cnn, X_train, Y_train, X_test, Y_test, bat_, epo_)

    # Save the model
    my_cnn.save('cifar10.h5')
