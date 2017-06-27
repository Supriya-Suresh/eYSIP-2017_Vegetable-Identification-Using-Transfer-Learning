from helper import *

import tensorflow as tf
import prettytensor as pt
import time
from datetime import timedelta
import os

from dataset import load_cached

dataset = load_cached(cache_path='veg.pkl', in_dir='./veg')
num_classes = dataset.num_classes
class_names = dataset.class_names

image_paths_train, cls_train, labels_train = dataset.get_training_set()
image_paths_test, cls_test, labels_test = dataset.get_test_set()

print("Size of:")
print("- Training-set:\t\t{}".format(len(image_paths_train)))
print("- Test-set:\t\t{}".format(len(image_paths_test)))
import inception

inception.maybe_download()
model = inception.Inception()

data_dir = "./veg/"
file_path_cache_train = os.path.join(data_dir, 'inception-veg-train.pkl')
file_path_cache_test = os.path.join(data_dir, 'inception-veg-test.pkl')

print("Processing Inception transfer-values for training-images ...")

# If transfer-values have already been calculated then reload them,
# otherwise calculate them and save them to a cache-file.
transfer_values_train = inception.transfer_values_cache(cache_path=file_path_cache_train,
                                                        image_paths=image_paths_train,
                                                        model=model)

print("Processing Inception transfer-values for test-images ...")

# If transfer-values have already been calculated then reload them,
# otherwise calculate them and save them to a cache-file.
transfer_values_test = inception.transfer_values_cache(cache_path=file_path_cache_test,
                                                       image_paths=image_paths_test,
                                                       model=model)


# plot_transfer_values(image_paths_train, transfer_values_train, 545)
# plot_transfer_values(image_paths_train, transfer_values_train, 546)

is_Training = True
# Useful for visualizing if transfer values can be grouped into classes
# from sklearn.decomposition import PCA

# pca = PCA(n_components=2)
# transfer_values = transfer_values_train
# cls = cls_train
# transfer_values_reduced = pca.fit_transform(transfer_values)
# plot_scatter(transfer_values_reduced,num_classes,cls)

# from sklearn.manifold import TSNE

# pca = PCA(n_components=50)
# transfer_values_50d = pca.fit_transform(transfer_values)
# tsne = TSNE(n_components=2)
# transfer_values_reduced = tsne.fit_transform(transfer_values_50d)
# plot_scatter(transfer_values_reduced,num_classes, cls)


transfer_len = model.transfer_len
x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true,dimension=1)

x_pretty = pt.wrap(x)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred,loss = x_pretty.\
        fully_connected(size=4096,name='layer_fc1').\
        fully_connected(size=2048,name='layer_fc2').\
        dropout(keep_prob=0.5,phase=is_Training).\
        fully_connected(size=1024,name='layer_fc3').\
        softmax_classifier(num_classes=num_classes,labels=y_true)

global_step = tf.Variable(initial_value=0,name='global_step',trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=(1e-5)).minimize(loss, global_step)
y_pred_cls = tf.argmax(y_pred,dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

saver = tf.train.Saver()

save_dir = 'checkpoints/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

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
    is_Training = True
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
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the indices for the incorrectly classified images.
    idx = np.flatnonzero(incorrect)

    # Number of images to select, max 9.
    n = min(len(idx), 9)

    # Randomize and select n indices.
    idx = np.random.choice(idx,
                           size=n,
                           replace=False)

    # Get the predicted classes for those images.
    cls_pred = cls_pred[idx]

    # Get the true classes for those images.
    cls_true = cls_test[idx]

    # Load the corresponding images from the test-set.
    # Note: We cannot do image_paths_test[idx] on lists of strings.
    image_paths = [image_paths_test[i] for i in idx]
    images = load_images(image_paths)

    # Plot the images.
    plot_images(images=images, cls_true=cls_true,class_names=class_names,cls_pred=cls_pred)

# Import a function from sklearn to calculate the confusion-matrix.
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(num_classes):
        # Append the class-name to each line.
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))


batch_size = 256


def predict_cls(transfer_values, labels, cls_true):
    is_Training = False
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


def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    is_Training = False
    # For all the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    correct, cls_pred = predict_cls_test()

    # Classification accuracy and the number of correct classifications.
    acc, num_correct = classification_accuracy(correct)

    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


def predict_class_of_image(image_path):
    transfer_value = [model.transfer_values(image_path=image_path)]    
    feed_dict = {x: transfer_value}
    classification = session.run(y_pred_cls,feed_dict)
    class_percent = session.run(y_pred,feed_dict)
    #return classification
    return dataset.class_names[classification[0]],(str(max(class_percent[0]*100)) + '%')


num_iterations = 10000
print("Before Training")
print_test_accuracy(False, False)
optimize(num_iterations=num_iterations)

# To restore previously saved model
#saver.restore(sess=session,save_path=save_path)
print("After training")
print_test_accuracy(False, True)


model.close()
session.close()

