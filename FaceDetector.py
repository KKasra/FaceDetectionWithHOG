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


def add_padding(image, sample_dims):
    padded_current_image = np.zeros((sample_dims[0] + image.shape[0],
                                     sample_dims[1] + image.shape[1],
                                     3))
        
    shift = np.array(padded_current_image.shape[:2]) - np.array(image.shape[:2])
    shift = (shift / 2).astype(int)
    padded_current_image[shift[0]:image.shape[0]+shift[0], shift[1]:image.shape[1]+shift[1], :] = image

    return padded_current_image, shift


def detect_faces(image):

    parameters = json.load(open('parameters.json'))

    number_of_orientations = parameters['number_of_orientations']
    pixels_per_cell = parameters['pixels_per_cell']
    cells_per_block = parameters['cells_per_block']


    extract_feature_vector = lambda img: hog(img,orientations=number_of_orientations, 
                                            pixels_per_cell=pixels_per_cell,
                                            cells_per_block=cells_per_block,
                                            channel_axis=2)



    sample_dims = parameters['sample_dims']
    strides = parameters['sliding_window_strides']

    levels = parameters['levels']

    svm_classifier = pickle.load(open('classifier.pickle', 'rb'))
    
    
    
    bounding_boxes = []
    for i in range(len(levels)):
        resize_factor = levels[i]
        stride = strides[i]
        print("level {i}".format(i=i))

        # Padding the image 
        current_image = cv.resize(image, (0,0), fx = resize_factor, fy = resize_factor)
        padded_current_image, shift = add_padding(current_image, sample_dims=sample_dims)
        del current_image

        tmp_bounding_boxes = []
        feature_vectors = []
        for b_x in range(0, padded_current_image.shape[0] - sample_dims[0], stride):
            for b_y in range(0, padded_current_image.shape[1] - sample_dims[1], stride):
                sample = padded_current_image[b_x:b_x+sample_dims[0], b_y:b_y+sample_dims[1]]
                feature_vector = extract_feature_vector(sample)
                
                feature_vectors.append(feature_vector)   
                del feature_vector

                tmp_bounding_boxes.append([(b_x - shift[0]) / resize_factor, (b_y - shift[1]) / resize_factor, 
                                           sample_dims[0] / resize_factor, sample_dims[1] / resize_factor, 
                                           0])
                
        del padded_current_image
        
        # Filter the boxes

        prediction = svm_classifier.predict(feature_vectors)

        if prediction.sum() == 0:
            continue

        tmp_bounding_boxes = np.array(tmp_bounding_boxes)
        feature_vectors = np.array(feature_vectors)
    
        tmp_bounding_boxes = tmp_bounding_boxes[np.where(prediction == 1)[0]]
        feature_vectors = feature_vectors[np.where(prediction == 1)[0]]

        tmp_bounding_boxes[:,-1] = svm_classifier.decision_function(feature_vectors)

        bounding_boxes = bounding_boxes + [v for v in tmp_bounding_boxes]

        del feature_vectors
        del prediction
        del tmp_bounding_boxes
    

    return bounding_boxes
    

def eliminate_redundante_boxes(bounding_boxes):

    parameters = json.load(open('parameters.json'))

    accepting_threshold = parameters['decision_function_threshold']

    bounding_boxes = np.array(bounding_boxes)
    bounding_boxes = bounding_boxes[np.argsort(-bounding_boxes[:, 4])]


    accepted_bounding_boxes = []

    for i in range(bounding_boxes.shape[0]):
        if bounding_boxes[i, 4] < accepting_threshold:
            break
        surface_area = bounding_boxes[i, 2] * bounding_boxes[i, 3]
        flag = True
        for b in accepted_bounding_boxes:
            # calculate surface area of intersection
            left = max(bounding_boxes[i, 0], b[0])
            right = min(bounding_boxes[i, 0] + bounding_boxes[i, 2], b[0] + b[2])
            top = max(bounding_boxes[i, 1], b[1])
            bottom = min(bounding_boxes[i, 1] + bounding_boxes[i, 3], b[1] + b[3])


            if left < right and top < bottom:
                intersection_area = (right - left) * (bottom - top)
                
                if (intersection_area / (b[2] * b[3] + surface_area - intersection_area)) > 0.6:
                    flag = False
                    break
                if intersection_area / surface_area > .6:
                    flag = False
                    break
                if intersection_area / (b[2] * b[3]) > .8:
                    flag = False
                    break

        if flag:
            accepted_bounding_boxes.append(bounding_boxes[i])
        


    return accepted_bounding_boxes

def get_detection_image(image):
    tmp = image.copy()
    detection_bounding_boxes = detect_faces(tmp)
    bounding_boxes = eliminate_redundante_boxes(detection_bounding_boxes)
    
    
    tmp_with_boxes = tmp.copy()

    for bb in bounding_boxes:
        r, b, g = np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)
        cv.rectangle(tmp_with_boxes, (int(bb[1]), int(bb[0])), (int(bb[1]+bb[3]), int(bb[0]+bb[2])),(r,g,b), 2)
        cv.putText(tmp_with_boxes, str(bb[4])[:5], (int(bb[1]), int(bb[0])), cv.FONT_HERSHEY_SIMPLEX, .5, (r,g,b), 2)

    


    return tmp_with_boxes

if __name__ == '__main__':
    input_image = cv.imread(input('Enter the input image path: '))
    output_image = get_detection_image(input_image)
    cv.imwrite(input('Enter the output image path: '), output_image)