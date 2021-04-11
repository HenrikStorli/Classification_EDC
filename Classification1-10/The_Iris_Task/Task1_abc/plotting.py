from import_utilities import *

#  Task 2
feature_labels = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
color_labels = ['red','green','blue','orange']
def setosa_hist_plots():
    for i in range(len(feature_labels)):
        plt.subplot(2,2,i+1)
        plt.title(feature_labels[i])
        plt.xlabel("Length [cm]")
        plt.ylabel("Samples")
        plt.xlim((0,8))
        plt.ylim((0,17))
        plt.hist(setosa[:,i], bins=10, color=color_labels[i])

    plt.subplots_adjust(hspace=0.5)

    plt.suptitle("Setosa")

    plt.show()

def versicolor_hist_plots():
    for i in range(len(feature_labels)):
        plt.subplot(2, 2, i+1)
        plt.title(feature_labels[i])
        plt.xlabel("Length [cm]")
        plt.ylabel("Samples")
        plt.xlim((0, 8))
        plt.ylim((0, 17))
        plt.hist(versicolor[:, i], bins=10, color=color_labels[i])

    plt.subplots_adjust(hspace=0.5)

    plt.suptitle("Versicolor")

    plt.show()


def virginica_hist_plots():
    for i in range(len(feature_labels)):
        plt.subplot(2, 2, i+1)
        plt.title(feature_labels[i])
        plt.xlabel("Length [cm]")
        plt.ylabel("Samples")
        plt.xlim((0, 8))
        plt.ylim((0, 17))
        plt.hist(virginica[:, i], bins=10, color=color_labels[i])

    plt.subplots_adjust(hspace=0.5)

    plt.suptitle("Virginica")

    plt.show()