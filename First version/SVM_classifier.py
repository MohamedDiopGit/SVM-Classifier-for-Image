from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib notebook
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split

import skimage
from skimage.io import imread
from skimage.transform import resize


def load_image_files(container_path, dimension=(64, 64)):
    """
    Load image files with categories as subfolder names 
    which performs like scikit-learn sample dataset
    
    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category
    dimension : tuple
        size to which image are adjusted to
        
    Returns
    -------
    Bunch
    """
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    skipped = 0
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            try:
                img = skimage.io.imread(file)
                img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
                flat_data.append(img_resized.flatten()) 
                images.append(img_resized)
                target.append(i)
            except ValueError:
                skipped = skipped + 1
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr),skipped


# DATASET PATH

image_dataset, skipped = load_image_files("Datasets/Cucumber")
# FORMAT DU DATASET :
#       PLANTE
#             HEALTHY
#             MALADIE1
#             MALEDIE2
print("Skipped:",skipped)

X_train, X_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.3,random_state=109)

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
svc = svm.SVC()
clf = GridSearchCV(svc, param_grid)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


print("Classification report for - \n{}:\n{}\n".format(
    clf, metrics.classification_report(y_test, y_pred)))


imgTest = skimage.io.imread("Datasets/images/pizza/image_0001.jpg")
dimension=(64,64)
imgTest = resize(imgTest, dimension, anti_aliasing=True, mode='reflect')
imgTest = imgTest.flatten()
myTest = []
myTest.append(imgTest)


pred = clf.predict(myTest)
indice = pred[0]
print(image_dataset.target_names[indice])


# save the model to disk
import pickle
filename = 'svm_model.sav'
pickle.dump(clf, open(filename, 'wb'))


# load the model from disk
import pickle
filename = 'svm_model.sav'
clf = pickle.load(open(filename, 'rb'))
result = clf.score(X_test, y_test)
print(result)