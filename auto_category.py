from sklearn.cluster import KMeans
import imutils
from imutils import paths
import numpy as np
import argparse
import cv2

def describe(image):
    # convert the image to the L*a*b* color space, compute a histogram,
    # and normalize it
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [0, 1, 2], None, [8,8,8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist).flatten()
    # return the histogram
    return hist

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to the input dataset directory")
ap.add_argument("-k", "--clusters", type=int, default=2, help="# of clusters to generate")
args = vars(ap.parse_args())

# initialize the image descriptor along with the image matrix
data = []

# grab the image paths from the dataset directory
imagePaths = list(paths.list_images(args["dataset"]))
imagePaths = np.array(sorted(imagePaths))

# loop over the input dataset of images
for imagePath in imagePaths:
        # load the image, describe the image, then update the list of data
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width = 350)
        hist = describe(image)
        data.append(hist)

# cluster the color histograms
clt = KMeans(n_clusters=args["clusters"], random_state=42)
labels = clt.fit_predict(data)

# loop over the unique labels
for label in np.unique(labels):
        # grab all image paths that are assigned to the current label
        labelPaths = imagePaths[np.where(labels == label)]

        # loop over the image paths that belong to the current label
        for (i, path) in enumerate(labelPaths):
                # load the image and display it
                image = cv2.imread(path)
                image = imutils.resize(image, width = 250)
                cv2.imshow("Cluster {}, Image #{}".format(label + 1, i + 1), image)

        # wait for a keypress and then close all open windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

