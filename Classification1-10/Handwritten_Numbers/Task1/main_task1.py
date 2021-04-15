from classifier import *
from import_utilities import *

# Import data from file
    # matrix: Full training set
    # matrix: Full test set
    # vector: Training lables
    # vector: Test labels

subset_size = 100
test_images, test_labels, train_images, train_labels = import_all_subset(subset_size)

# Split training set and test set into smaller sets

a = 1
# Classify

guessed_class_vector = NN_classifier(test_images, train_images, train_labels)

## Confusion matrix
C = 10
confusion_matrix = confusion_matrix(guessed_class_vector, test_labels, C)
print("Confusion matrix for test set: \n", confusion_matrix)

## Error rate

error_rate_test_set = error_rate(guessed_class_vector, test_labels)
print("Errors test: ", error_rate_test_set, '\n', round(error_rate_test_set * 100, 1), '% \n')
a = 1
# 1b)

# Find some of the misclassified pictures

# Plot some of the misclassified pictures

# Finds some correctly classified pictures

# Plot some correctly classified pictures



