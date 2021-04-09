from import_utilities import *

#  Task 2

def setosa_hist_plots():

    plt.subplot(2,2,1)
    plt.title("Sepal length")
    plt.xlabel("Length [cm]")
    plt.ylabel("Samples")
    plt.xlim((0,8))
    plt.ylim((0,17))
    plt.hist(setosa[:,0], bins=10, color='red')

    plt.subplot(2,2,2)
    plt.title("Sepal width")
    plt.xlabel("Width [cm]")
    plt.ylabel("Samples")
    plt.xlim((0, 8))
    plt.ylim((0, 17))
    plt.hist(setosa[:,1], bins=10, color='blue')

    plt.subplot(2,2,3)
    plt.title("Petal length")
    plt.xlabel("Length [cm]")
    plt.ylabel("Samples")
    plt.xlim((0, 8))
    plt.ylim((0, 17))
    plt.hist(setosa[:,2], bins=10, color='green')

    plt.subplot(2,2,4)
    plt.title("Petal width")
    plt.xlabel("Width [cm]")
    plt.ylabel("Samples")
    plt.xlim((0, 8))
    plt.ylim((0, 17))
    plt.hist(setosa[:,3], bins=10, color='orange')

    plt.subplots_adjust(hspace=0.5)

    plt.suptitle("Setosa")

    plt.show()

def versicolor_hist_plots():
    plt.subplot(2,2,1)
    plt.title("Sepal length")
    plt.xlabel("Length [cm]")
    plt.ylabel("Samples")
    plt.xlim((0, 8))
    plt.ylim((0, 17))
    plt.hist(versicolor[:,0], bins=10, color='red')

    plt.subplot(2,2,2)
    plt.title("Sepal width")
    plt.xlabel("Width [cm]")
    plt.ylabel("Samples")
    plt.xlim((0, 8))
    plt.ylim((0, 17))
    plt.hist(versicolor[:,1], bins=10, color='blue')

    plt.subplot(2,2,3)
    plt.title("Petal length")
    plt.xlabel("Length [cm]")
    plt.ylabel("Samples")
    plt.xlim((0, 8))
    plt.ylim((0, 17))
    plt.hist(versicolor[:,2], bins=10, color='green')

    plt.subplot(2,2,4)
    plt.title("Petal width")
    plt.xlabel("Width [cm]")
    plt.ylabel("Samples")
    plt.xlim((0, 8))
    plt.ylim((0, 17))
    plt.hist(versicolor[:,3], bins=10, color='orange')

    plt.subplots_adjust(hspace=0.5)

    plt.suptitle("Versicolor")

    plt.show()


def virginica_hist_plots():
    plt.subplot(2,2,1)
    plt.title("Sepal length")
    plt.xlabel("Length [cm]")
    plt.ylabel("Samples")
    plt.xlim((0, 8))
    plt.ylim((0, 17))
    plt.hist(virginica[:,0], bins=10, color='red')

    plt.subplot(2,2,2)
    plt.title("Sepal width")
    plt.xlabel("Width [cm]")
    plt.ylabel("Samples")
    plt.xlim((0, 8))
    plt.ylim((0, 17))
    plt.hist(virginica[:,1], bins=10, color='blue')

    plt.subplot(2,2,3)
    plt.title("Petal length")
    plt.xlabel("Length [cm]")
    plt.ylabel("Samples")
    plt.xlim((0, 8))
    plt.ylim((0, 17))
    plt.hist(virginica[:,2], bins=10, color='green')

    plt.subplot(2,2,4)
    plt.title("Petal width")
    plt.xlabel("Width [cm]")
    plt.ylabel("Samples")
    plt.xlim((0, 8))
    plt.ylim((0, 17))
    plt.hist(virginica[:,3], bins=10, color='orange')

    plt.subplots_adjust(hspace=0.5)

    plt.suptitle("Virginica")

    plt.show()