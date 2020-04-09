#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    Function that builds an inception block as described in Going Deeper with
    Convolutions (2014)

    All convolutions inside the inception block should use a rectified linear
    activation (ReLU)

    Args:
        - A_prev:   is the output of the previous layer
        - filters   is a tuple or list containing F1, F3R, F3,F5R, F5, FPP,
    """
    X_p = K.applications.vgg16.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, num_classes=10)
    return (X_p, Y_p)


def load_dataset():
    """
    Function that loads a CIFAR 10 dataset and gnerate the Xs Ys sets

    Args: None

    Returns: A dataset divided in training and test sets
    """
    # 3. Import the CIFAR-10 dataset, normalize data by dividing it to 255
    # X_train = X_train.astype('float32'), # X_test = X_test.astype('float32')
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


def define_model():
    """
    Function that generates the CNN model to work with CIFAR 10 dataset

    Args: None

    Returns: A model
    """

    # Importing VGG16 dataset, allowing to use any size
    # vgg = K.applications.vgg16.VGG16(include_top=False)

    vgg = K.applications.vgg16.VGG16(include_top=True,
                                     weights=True,
                                     input_shape=(32, 32, 3))

    # Creating my own copy of the vgg arquitecture
    my_cnn = K.Sequential()

    for lyr in vgg.layers:
        my_cnn.add(lyr)

    # Removing the last activation layer
    my_cnn.pop()
    my_cnn.summary()

    # Freezing the layers of the original CNN
    for layer in my_cnn.layers:
        layer.trainable = False

    #X_train = X_train.reshape((X_train.shape[0], 3072))
    #X_test = X_test.reshape((X_test.shape[0], 3072))

    # Inserting some activation layers to my CNN
    my_cnn.add(K.layers.Conv2D(32, (3, 3), padding='same'))
    my_cnn.add(K.layers.BatchNormalization())
    my_cnn.add(K.layers.Activation('elu'))

    my_cnn.add(K.layers.Conv2D(32, (3, 3), padding='same'))
    my_cnn.add(K.layers.BatchNormalization())
    my_cnn.add(K.layers.Activation('elu'))

    my_cnn.add(K.layers.MaxPooling2D(pool_size=(2, 2)))
    my_cnn.add(K.layers.Dropout(0.2))

    my_cnn.add(K.layers.Conv2D(64, (3, 3), padding='same'))
    my_cnn.add(K.layers.BatchNormalization())
    my_cnn.add(K.layers.Activation('elu'))

    my_cnn.add(K.layers.Conv2D(64, (3, 3), padding='same'))
    my_cnn.add(K.layers.BatchNormalization())
    my_cnn.add(K.layers.Activation('elu'))

    my_cnn.add(K.layers.MaxPooling2D(pool_size=(2, 2)))
    my_cnn.add(K.layers.Dropout(0.2))

    #my_cnn.add(K.layers.Conv2D(128, (3, 3)))
    my_cnn.add(K.layers.Conv2D(128, (1, 1)))

    my_cnn.add(K.layers.BatchNormalization())
    my_cnn.add(K.layers.Activation('elu'))

    # my_cnn.add(K.layers.Conv2D(128, (3, 3)))
    my_cnn.add(K.layers.Conv2D(128, (1, 1)))

    my_cnn.add(K.layers.BatchNormalization())
    my_cnn.add(K.layers.Activation('elu'))

    #my_cnn.add(K.layers.MaxPooling2D(pool_size=(2, 2)))
    my_cnn.add(K.layers.MaxPooling2D(pool_size=(1, 1)))

    my_cnn.add(K.layers.Dropout(0.5))
    my_cnn.add(K.layers.Flatten())
    my_cnn.add(K.layers.Dense(10, activation='softmax'))
    my_cnn.summary()

    return my_cnn


def define_model_VGG():
    """
    Function that generates the CNN model to work with CIFAR 10 dataset

    Args: None

    Returns: A model compiled VGG model
    """

    model = K.Sequential()
    model.add(K.layers.Conv2D(32,
                              (3, 3),
                              activation='relu',
                              kernel_initializer='he_uniform',
                              padding='same',
                              input_shape=(32, 32, 3)))
    model.add(K.layers.Conv2D(32,
                              (3, 3),
                              activation='relu',
                              kernel_initializer='he_uniform',
                              padding='same'))
    model.add(K.layers.MaxPooling2D((2, 2)))
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(128,
                             activation='relu',
                             kernel_initializer='he_uniform'))
    model.add(K.layers.Dense(10, activation='softmax'))

    # compile model
    opt = K.optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


# plot diagnostic learning curves
def summarize_diagnostics(history):
    from matplotlib import pyplot

    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(
        history.history['val_accuracy'],
        color='orange',
        label='test')
    # save plot to file
    filename = "0-transfer"
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()

# *********************************MAIN***************************************
# Ref: https://bit.ly/2Rj2z1u. Python Deep Learning. 2nd Ed. Pag 118
# Ref: https://bit.ly/2xZcXVD

# load dataset
X_train, Y_train, X_test, Y_test = load_dataset()

"""
# 4. Use augmentation types:
#   - We-ll allow rotation of up to 90 degrees, horizontal flip, horizontal
#     and vertical shift of the data
#   - We'll standardize the training data (featurewise_center and
#     featurewise_std_normalization). Because the mean and standard deviation
#     are computed over the whole data set, we need to call the
#     data_generator.fit(X_train) method
#   - We need to apply the training standardization over the test set.
#     ImageDataGenerator will generate a stream of augmented images during
#     training

# image generator:

ig = K.preprocessing.image.ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.1,
    height_shift_range=0.1,
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True)
ig.fit(X_train)
K.applications.vgg16.preprocess_input(X_train)

# standardize the test set
for i in range(len(X_test)):
    X_test[i] = ig.standardize(X_test[i])
"""

# define model
my_cnn = define_model_VGG()

# fit model
history = my_cnn.fit(X_train,
                     Y_train,
                     epochs=27,
                     batch_size=64,
                     validation_data=(X_test, Y_test), verbose=1)

# evaluate model
_, acc = my_cnn.evaluate(X_test, Y_test, verbose=1)
print('> %.3f' % (acc * 100.0))

# learning curves
summarize_diagnostics(history)


# 6. Next, we'll define the optimizer, in this case, Adam
my_cnn.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

# 7. Then, we'll train the network. Because of the data augmentation, we'll
# use the model.fit_generator method. Our generator is the ImageDataGenerator,
# we defined earlier. We'll use the test set as validation_data. In this way
# we'll know our actual performance after each epoch.

my_cnn.fit_generator(generator=data_generator.flow(x=X_train,
                                                   y=Y_train,
                                                   batch_size=batch_size),
                     steps_per_epoch=len(X_train) // batch_size,
                     epochs=100,
                     validation_data=(X_test, Y_test),
                     workers=4)
