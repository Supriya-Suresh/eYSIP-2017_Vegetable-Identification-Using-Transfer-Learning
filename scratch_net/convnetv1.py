import tensorflow as tf
import matplotlib.pyplot as plt
from helper import *
import numpy as np
import math
import time
import os
from datetime import timedelta

from dataset import load_cached

dataset = load_cached(cache_path='veg.pkl', in_dir='./veg')

image_paths_train, cls_train, labels_train = dataset.get_training_set()
image_paths_test, cls_test, labels_test = dataset.get_test_set()

labels_test_cls = np.argmax(labels_test,axis=1)

img_size = 300
img_shape = (img_size,img_size)
num_channels = 3
img_size_flat = img_size * img_size * num_channels

num_classes = dataset.num_classes
class_names = dataset.class_names


print("Size of:")
print("- Training-set:\t\t{}".format(len(image_paths_train)))
print("- Test-set:\t\t{}".format(len(image_paths_test)))

# Get the true classes for those images.
# cls_true = cls_test[0:9]

# # Plot the images and labels using our helper-function above.
# plot_images(images=images_test[0:9], cls_true=cls_true, smooth=True,class_names=class_names)

x = tf.placeholder(tf.float32,shape=[None, img_size_flat],name='x')
x_image = tf.reshape(x,shape=[-1,img_size,img_size,num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# Hyper-Parameters
learning_rate = 1e-4
num_iterations = 5000

# Convolutional Layer 1
num_filters_layer_1 = 32
kernel_size_layer_1 = 6

pool_size_layer_1 = 2
pool_strides_layer_1 = 2

# Convolutional Layer 2
num_filters_layer_2 = 64
kernel_size_layer_2 = 6

pool_size_layer_2 = 2
pool_strides_layer_2 = 2

# Convolutional Layer 3
num_filters_layer_3 = 128
kernel_size_layer_3 = 6

pool_size_layer_3 = 2
pool_strides_layer_3 = 2

# Convolutional Layer 4
num_filters_layer_4 = 256
kernel_size_layer_4 = 6

pool_size_layer_4 = 2
pool_strides_layer_4 = 2

# Convolutional Layer 5
num_filters_layer_5 = 512
kernel_size_layer_5 = 6

pool_size_layer_5 = 2
pool_strides_layer_5 = 2

# Fully Connected Layer
fcl_num_units = 1024
dropout_rate = 0.8

is_training = True


net = x_image

# Convolutional Layer 1
net = tf.layers.conv2d(inputs=net, filters=num_filters_layer_1,
                       kernel_size=kernel_size_layer_1,
                       name='layer_conv1', padding='same',
                       activation=tf.nn.relu)

layer_conv1 = net

net = tf.layers.max_pooling2d(inputs=net, pool_size=pool_size_layer_1,
                              name='pool1',
                              strides=pool_strides_layer_1)


# Convolutional Layer 2
net = tf.layers.conv2d(inputs=net, filters=num_filters_layer_2,
                       kernel_size=kernel_size_layer_2,
                       name='layer_conv2', padding='same',
                       activation=tf.nn.relu)

layer_conv2 = net

net = tf.layers.max_pooling2d(inputs=net, pool_size=pool_size_layer_2,
                              name='pool2',
                              strides=pool_strides_layer_2)

# Convolutional Layer 3
net = tf.layers.conv2d(inputs=net, filters=num_filters_layer_3,
                       kernel_size=kernel_size_layer_3,
                       name='layer_conv3', padding='same',
                       activation=tf.nn.relu)

layer_conv3 = net

net = tf.layers.max_pooling2d(inputs=net, pool_size=pool_size_layer_3,
                              name='pool3',
                              strides=pool_strides_layer_3)

# Convolutional Layer 4
net = tf.layers.conv2d(inputs=net, filters=num_filters_layer_4,
                       kernel_size=kernel_size_layer_4,
                       name='layer_conv4', padding='same',
                       activation=tf.nn.relu)

layer_conv4 = net

net = tf.layers.max_pooling2d(inputs=net, pool_size=pool_size_layer_4,
                              name='pool4',
                              strides=pool_strides_layer_4)

# Convolutional Layer 5
net = tf.layers.conv2d(inputs=net, filters=num_filters_layer_5,
                       kernel_size=kernel_size_layer_5,
                       name='layer_conv5', padding='same',
                       activation=tf.nn.relu)

layer_conv5 = net

net = tf.layers.max_pooling2d(inputs=net, pool_size=pool_size_layer_5,
                              name='pool5',
                              strides=pool_strides_layer_5)

net = tf.contrib.layers.flatten(net)

# Fully Connected Layer
net = tf.layers.dense(inputs=net, name='layer_fc1',
                      units=fcl_num_units, activation=tf.nn.relu)

net = tf.layers.dropout(inputs=net,rate=dropout_rate,training=is_training,name='dropout')

# Output Layer
net = tf.layers.dense(inputs=net, name='layer_fc_out',
                      units=num_classes, activation=None)

logits = net

y_pred = tf.nn.softmax(logits=logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=logits)
loss = tf.reduce_mean(cross_entropy)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def get_weights_variable(layer_name):
    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('kernel')
    return variable

weights_conv1 = get_weights_variable(layer_name='layer_conv1')
weights_conv2 = get_weights_variable(layer_name='layer_conv2')
weights_conv3 = get_weights_variable(layer_name='layer_conv3')

saver = tf.train.Saver()
save_dir = 'checkpoints_scratch_2/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, 'saved')

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

session.run(tf.global_variables_initializer())

total_iterations = 0


train_batch_size = 64

from scipy.ndimage import imread


def load_images(image_paths):
    images = np.array([imread(path,mode='RGB',flatten=False) for path in image_paths])
    images = np.array(images)
    return images


def random_batch():
    # Number of images (transfer-values) in the training-set.
    num_images = len(image_paths_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random x and y-values.
    paths = [image_paths_train[a] for a in idx]
    x_batch = load_images(paths)
    y_batch = np.asarray([labels_train[a] for a in idx])

    return x_batch, y_batch

total_iterations = 0


def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations
    is_training = True
    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = random_batch()
        x_batch = x_batch.reshape(train_batch_size,img_size_flat)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            saver.save(sess=session,save_path=save_path)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

#Split the data-set in batches of this size to limit RAM usage.
batch_size = 64


def predict_cls(images, labels, cls_true):
    is_training = False
    # Number of images.
    num_images = len(images)

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

        x_val = load_images(images[i:j])
        y_val = labels[i:j]
        
        shape = (batch_size,img_size,img_size,num_channels)
        if(x_val.shape != shape):
            break
        x_val = x_val.reshape(-1,img_size_flat)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: x_val,
                     y_true: y_val}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j
        
    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred



def predict_cls_test():
    return predict_cls(images = image_paths_test,
                       labels = labels_test,
                       cls_true = cls_test)


def classification_accuracy(correct):
    return correct.mean(), correct.sum()


from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cls_pred):

    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

    for i in range(num_classes):
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)

    class_numbers = [" ({0})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))


def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    correct, cls_pred = predict_cls_test()
    acc, num_correct = classification_accuracy(correct)

    # Number of images being classified.
    num_images = len(correct)

    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


print_test_accuracy(show_example_errors=False,
                    show_confusion_matrix=True)

optimize(num_iterations=num_iterations)
#saver.restore(sess=session,save_path=save_path)

print_test_accuracy(show_example_errors=False,
                    show_confusion_matrix=True)
