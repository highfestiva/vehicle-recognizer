#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from time import time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC



def load_vehicles():
    from glob import glob
    from io import BytesIO
    from PIL import Image
    import random
    import re
    import struct
    images = []
    targets = []
    for fn in glob('data/*.jpg'):
        bmp = BytesIO()
        Image.open(fn).convert('RGB').save(bmp, 'BMP')
        bmp = bmp.getvalue()
        w = struct.unpack("<L", bmp[18:22])[0]
        h = struct.unpack("<L", bmp[22:26])[0]
        img = [b for i in range(h-1,-1,-1) for b in bmp[54+i*w*3:54+(i+1)*w*3:3]]
        images.append(img)
        targets.append(re.sub(r'\d.*','',fn.replace('\\','/').replace('data/','')))
    # Shuffle 'em.
    its = list(zip(images,targets))
    random.shuffle(its)
    images  = [i for i,t in its]
    targets = [t for i,t in its]
    return w, h, images, targets


w,h,images,targets = load_vehicles()
X = np.array(images)
n_samples = X.shape[0]
target_names = sorted(set(targets))
y = np.array([target_names.index(t) for t in targets])
target_names = np.array(target_names)

n_features = X.shape[1]
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# #############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# #############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 50

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


# #############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
#param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
#clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = SVC(kernel='rbf', class_weight='balanced', C=1e3, gamma=0.01)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
#print("Best estimator found by grid search:")
#print(clf.best_estimator_)


# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


# #############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=4, n_col=5):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        tit,match = titles[i]
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray if match else plt.cm.Reds)
        plt.title(tit, size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name), pred_name==true_name

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = [("eigenface %d"%i,True) for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()
