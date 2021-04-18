from classifier import *
from import_utilities import *
from clustering import *

import matplotlib.pyplot as plt


# Import data from file
    # matrix: Full training set
    # matrix: Full test set
    # vector: Training lables
    # vector: Test labels

# Split training set and test set into smaller sets
subset_size = 100

# Read values into variables
test_images, \
test_labels, \
train_images, \
train_labels \
    = import_all_subset(subset_size)

sorted_images = sort_training_images_into_classes(train_images, train_labels)

# Classify
guessed_class_vector = NN_classifier(test_images, train_images, train_labels)

## Confusion matrix
C = 10
confusion_matrix = confusion_matrix(guessed_class_vector, test_labels, C)
print("Confusion matrix for test set: \n", confusion_matrix)

## Error rate
error_rate_test_set = error_rate(guessed_class_vector, test_labels)
print("Errors test: ", error_rate_test_set, '\n', round(error_rate_test_set * 100, 1), '% \n')


# 1b)


# Find some of the misclassified pictures
misclassified_indexes = find_misclassified_indexes(guessed_class_vector, test_labels)
wrong_pic_idx = int(misclassified_indexes[1])
wrongly_classified_pic = np.reshape(test_images[wrong_pic_idx,:],(28,28))

# Plot some of the misclassified pictures
imgplot = plt.imshow(wrongly_classified_pic, cmap='gray')
plt.show()


# Finds some correctly classified pictures
correct_classified_indexes = find_correct_classified_indexes(guessed_class_vector, test_labels)
correct_pic_idx = int(correct_classified_indexes[0])
correct_classified_pic = np.reshape(test_images[correct_pic_idx,:],(28,28))

# Plot some correctly classified pictures
imgplot = plt.imshow(correct_classified_pic, cmap='gray')
plt.show()

a = 1 # For setting brakepoint

