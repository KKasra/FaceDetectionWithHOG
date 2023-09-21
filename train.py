# %%
import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
from sklearn.svm import SVC as svm
# from cv2 import ml_SVM as svm
from skimage.feature import hog
from sklearn.metrics import RocCurveDisplay as roc_display
from sklearn.metrics import PrecisionRecallDisplay as pr_display
import pickle
import json

# %% [markdown]
# ## load parameters

# %%
parameters = json.load(open('parameters.json'))
train_size = parameters['train_size']
validation_size = parameters['validation_size']
test_size = parameters['test_size']

# %% [markdown]
# ## load files

# %%
# iterate throw dataset directories

face_img_list = []
for d in os.listdir('lfw'):
    if d[0] == '.':
        continue
    for img in os.listdir('lfw/' + d):
        face_img_list.append('lfw/' + d + '/' + img)


non_face_img_list = []
size_list = []
for img in os.listdir('257.clutter'):   
    if img[0] == '.':
        continue 
    non_face_img_list.append('257.clutter/' + img)
    try:
        size_list.append(cv.imread('257.clutter/' + img).shape[:2])
    except:
        print(img)



# %% [markdown]
# ## take samples from non-face images

# %%
sample_dims = parameters['sample_dims']

non_face_sample_map = []

for i in range(len(non_face_img_list)):
    for b_x in range(0, size_list[i][0] - sample_dims[0], 50):
        for b_y in range(0, size_list[i][1] - sample_dims[1], 50):
            non_face_sample_map.append([non_face_img_list[i], b_x, b_y])

get_sample_from_map = lambda map: cv.imread(map[0])[map[1]:map[1]+sample_dims[0], map[2]:map[2]+sample_dims[1], ::-1]


# %% [markdown]
# ## choose parameters for HOG

# %%
number_of_orientations = parameters['number_of_orientations']
pixels_per_cell = parameters['pixels_per_cell']
cells_per_block = parameters['cells_per_block']


extract_feature_vector = lambda img: hog(img,orientations=number_of_orientations, 
                                            pixels_per_cell=pixels_per_cell,
                                            cells_per_block=cells_per_block,
                                            channel_axis=2)


img = cv.imread(face_img_list[20])[50:-50,50:-50,::-1]
tmp , hog_image = hog(img,orientations=number_of_orientations,
                        pixels_per_cell=pixels_per_cell,
                        cells_per_block=cells_per_block, 
                        channel_axis=2, visualize=True)

f, a = plt.subplots(1, 2)
f.set_figwidth(10)
f.set_figheight(5)
a[0].imshow(img)
a[1].imshow(hog_image)
print(tmp.shape)

# %% [markdown]
# ## extract features from samples

# %%
features = []
for i in range(len(face_img_list)):
    if i % 1000 == 0:
        print(i)
    img = cv.imread(face_img_list[i])[50:-50,50:-50]
    features.append(extract_feature_vector(img))



# %%
features = np.array(features)
np.save('positive_features.npy', features)

# %%
del features

# %%
features = []
for i in range(len(non_face_img_list)):
    if i % 100 == 0:
        print(i)
    img = cv.imread(non_face_img_list[i])
    for b_x in range(0, img.shape[0] - sample_dims[0], 50):
        for b_y in range(0, img.shape[1] - sample_dims[1], 50):
            features.append(extract_feature_vector(img[b_x:b_x+sample_dims[0], b_y:b_y+sample_dims[1]]))

features = np.array(features)
np.save('negative_features.npy', features)
del features


# %% [markdown]
# ## create training, validation, and test sets

# %%
positive_features = np.load('positive_features.npy')
positive_permutation = np.random.permutation(positive_features.shape[0])


negative_features = np.load('negative_features.npy')
negative_permutation = np.random.permutation(negative_features.shape[0])

np.save('validation_features.npy',np.concatenate([positive_features[positive_permutation[:validation_size]], 
                                                  negative_features[negative_permutation[:validation_size]]], axis=0))


np.save('test_features.npy',np.concatenate([positive_features[positive_permutation[validation_size:validation_size+test_size]], 
                                            negative_features[negative_permutation[validation_size:validation_size+test_size]]], axis=0))

positive_features = positive_features[positive_permutation[validation_size+test_size:validation_size+test_size+train_size]]
negative_features = negative_features[negative_permutation[validation_size+test_size:validation_size+test_size+train_size]]

np.save('train_features.npy', np.concatenate([positive_features, negative_features], axis=0))

del positive_features
del negative_features



# %% [markdown]
# ## train classifier

# %%
features = np.load('train_features.npy')
labels = np.concatenate([np.ones(train_size), np.zeros(train_size)], axis=0)

classifier = svm()

classifier.fit(features, labels)
# classifier.trainAuto(features[:10], cv.ml.ROW_SAMPLE, labels[:10])
del features

# %% [markdown]
# ## evaluate classifier

# %%
features = np.load('validation_features.npy')
labels = np.concatenate([np.ones(validation_size), np.zeros(validation_size)], axis=0)

prediction_on_validation_set = classifier.predict(features)

del features

# %%
(labels != prediction_on_validation_set).sum()

# %% [markdown]
# ## save classifier

# %%
pickle.dump(classifier, open('classifier.pickle', 'wb'))


# %% [markdown]
# ## draw performance plots

# %%
parameters = json.load(open('parameters.json'))

svm_classifier = pickle.load(open('classifier.pickle', 'rb'))

test_feature_vectors = np.load('test_features.npy')

ground_truth_labels = np.concatenate([np.ones(parameters['test_size']), np.zeros(parameters['test_size'])])

prediction = svm_classifier.predict(test_feature_vectors)


del test_feature_vectors

# %%

ROC_plot = roc_display.from_predictions(prediction, ground_truth_labels)

plt.savefig('res01.jpg')

# %%
precision_recall_plot = pr_display.from_predictions(prediction, ground_truth_labels)

plt.savefig('res02.jpg')

# %%



