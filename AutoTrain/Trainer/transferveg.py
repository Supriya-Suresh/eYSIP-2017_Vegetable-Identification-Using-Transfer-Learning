
import tensorflow as tf
import prettytensor as pt
import numpy as np
import time
from datetime import timedelta
import os
import shutil
import logging
from dataset import load_cached

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(filename='/home/sanket/Desktop/AutoTrain/Logs/log.txt',level=logging.DEBUG,filemode='a+',format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

# Server username
USERNAME = 'user'
SERVER_IP = '192.168.1.1'
SERVER_DIR = '/home/server/django/'

# Directories
data_dir = "/home/sanket/Desktop/AutoTrain/Images/"
save_dir = '/home/sanket/Desktop/AutoTrain/checkpoints'
cache_dir = '/home/sanket/Desktop/AutoTrain/cache/'
cache_path = os.path.join(save_dir,'veg.pkl')

all_dirs = [save_dir,cache_dir]

for directory in all_dirs:
    if os.path.exists(directory):
        shutil.rmtree(directory,ignore_errors=True)
    os.makedirs(directory)

dataset = load_cached(cache_path=cache_path, in_dir=data_dir)
num_classes = dataset.num_classes
class_names = dataset.class_names

image_paths_train, cls_train, labels_train = dataset.get_training_set()
image_paths_test, cls_test, labels_test = dataset.get_test_set()

size_info = "Size of: " + " Training-set: {}".format(len(image_paths_train)) + " Test-set: {}".format(len(image_paths_test))
logging.info(size_info)

import inception

inception.maybe_download()
model = inception.Inception()

file_path_cache_train = os.path.join(cache_dir, 'inception-veg-train.pkl')
file_path_cache_test = os.path.join(cache_dir, 'inception-veg-test.pkl')


# If transfer-values have already been calculated then reload them,
# otherwise calculate them and save them to a cache-file.
transfer_values_train = inception.transfer_values_cache(cache_path=file_path_cache_train,
                                                        image_paths=image_paths_train,
                                                        model=model)
logging.info("Done Processing Inception transfer-values for training-images!")

# If transfer-values have already been calculated then reload them,
# otherwise calculate them and save them to a cache-file.
transfer_values_test = inception.transfer_values_cache(cache_path=file_path_cache_test,
                                                       image_paths=image_paths_test,
                                                       model=model)
logging.info("Done Processing Inception transfer-values for test-images!")


transfer_len = model.transfer_len
x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true,dimension=1)

x_pretty = pt.wrap(x)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred,loss = x_pretty.\
        fully_connected(size=4096,name='layer_fc1').\
        fully_connected(size=2048,name='layer_fc2').\
        fully_connected(size=1024,name='layer_fc3').\
        softmax_classifier(num_classes=num_classes,labels=y_true)

global_step = tf.Variable(initial_value=0,name='global_step',trainable=False)
optimizer = tf.train.AdagradOptimizer(learning_rate=(1e-3)).minimize(loss, global_step)
y_pred_cls = tf.argmax(y_pred,dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

saver = tf.train.Saver()

save_path = os.path.join(save_dir, 'saved')

session = tf.Session()
session.run(tf.global_variables_initializer())

train_batch_size = 64


def random_batch():
    # Number of images (transfer-values) in the training-set.
    num_images = len(transfer_values_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random x and y-values.
    # We use the transfer-values instead of images as x-values.
    x_batch = transfer_values_train[idx]
    y_batch = labels_train[idx]

    return x_batch, y_batch


def optimize(num_iterations):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images (transfer-values) and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = random_batch()

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        # We also want to retrieve the global_step counter.
        i_global, _ = session.run([global_step, optimizer],
                                  feed_dict=feed_dict_train)

        # Print status to screen every 100 iterations (and last).
        if (i_global % 100 == 0) or (i == num_iterations - 1):
            # Calculate the accuracy on the training-batch.
            batch_acc = session.run(accuracy,
                                    feed_dict=feed_dict_train)

            saver.save(sess=session,save_path=save_path)

            # Print status.
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    logging.info("Time usage for training: " + str(timedelta(seconds=int(round(time_dif)))))


batch_size = 256


def predict_cls(transfer_values, labels, cls_true):
    # Number of images.
    num_images = len(transfer_values)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: transfer_values[i:j],
                     y_true: labels[i:j]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred


def predict_cls_test():
    return predict_cls(transfer_values = transfer_values_test,
                       labels = labels_test,
                       cls_true = cls_test)


def classification_accuracy(correct):
    # When averaging a boolean array, False means 0 and True means 1.
    # So we are calculating: number of True / len(correct) which is
    # the same as the classification accuracy.

    # Return the classification accuracy
    # and the number of correct classifications.
    return correct.mean(), correct.sum()


def print_test_accuracy():
    # For all the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    correct, cls_pred = predict_cls_test()

    # Classification accuracy and the number of correct classifications.
    acc, num_correct = classification_accuracy(correct)

    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    logging.info(msg.format(acc, num_correct, num_images))


num_iterations = 7000
logging.info("Before Training")
print_test_accuracy()
optimize(num_iterations=num_iterations)
logging.info("After training")
print_test_accuracy()

model.close()
session.close()

# import subprocess
# server_dest = USERNAME + '@' + SERVER_IP + ':' SERVER_DIR
# sync_checkpoints = 'rsync -auvz ' + save_dir + server_dest
# process = subprocess.Popen(sync_command.split(),stdout=subprocess.PIPE)
# output,error = process.communicate()

