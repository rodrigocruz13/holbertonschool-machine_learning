#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train,
                     Y_train,
                     X_valid,
                     Y_valid,
                     batch_size=32,
                     epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network model using mini-batch GD :

    Args:
        X numpy.ndarray of shape (m, nx) to shuffle
            - m is the number of data points
            - nx is the number of features
        Y is the second numpy.ndarray of shape (m, ny)
        to shuffle
            - m is the same number of data points as in X
            - ny is the number of features in Y
    Returns:
        The shuffled X and Y matrices
    """

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]

        bat_loss = tf.get_collection("loss")[0]
        accuracy = tf.get_collection("accuracy")[0]
        train_op = tf.get_collection("train_op")[0]

        feed_dict_t = {x: X_train, y: Y_train}
        feed_dict_v = {x: X_valid, y: Y_valid}

        float_i = X_train.shape[0] / batch_size
        int_i = int(float_i)

        extra = False
        if (float_i > int_i):
            int_i = int(float_i) + 1
            extra = True

        for epoch in range(epochs + 1):
            cost_t = sess.run(bat_loss, feed_dict_t)
            acc_t = sess.run(accuracy, feed_dict_t)
            cost_v = sess.run(bat_loss, feed_dict_v)
            acc_v = sess.run(accuracy, feed_dict_v)

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(cost_t))
            print("\tTraining Accuracy: {}".format(acc_t))
            print("\tValidation Cost: {}".format(cost_v))
            print("\tValidation Accuracy: {}".format(acc_v))

            if (epoch < epochs):
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
                for step in range(int_i):
                    start = step * batch_size
                    end = batch_size * (step + 1)
                    if step == int_i - 1 and extra:
                        end = int(batch_size * (step + float_i - int_i + 1))

                    feed_dict_mini = {x: X_shuffled[start: end],
                                      y: Y_shuffled[start: end]}
                    sess.run(train_op, feed_dict_mini)

                    if step != 0 and (step + 1) % 100 == 0:
                        print("\tStep {}:".format(step + 1))
                        cost_mini = sess.run(bat_loss, feed_dict_mini)
                        print("\t\tCost: {}".format(cost_mini))
                        acc_mini = sess.run(accuracy, feed_dict_mini)
                        print("\t\tAccuracy: {}".format(acc_mini))
        save_path = saver.save(sess, save_path)
    return save_path
