######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
# It draws boxes and scores around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
from os import walk
import xml.etree.ElementTree as ET
from google.protobuf import text_format
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_FILE = 'frozen_inference_graph.pb'
LABEL_FILE = 'label_map.pbtxt'
VERIFY_ROOT = "verify"
VERIFY_RESULT_FOLDER = os.path.join(VERIFY_ROOT, "result")
IMAGE_FOLDER = "images"
IMAGE_MAX = 600
# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(VERIFY_ROOT, MODEL_FILE)

# Path to label map file
PATH_TO_LABELS = os.path.join(VERIFY_ROOT, LABEL_FILE)

# Path to image
FOLDER_OF_IMAGES = os.path.join(VERIFY_ROOT,IMAGE_FOLDER)

# Number of classes the object detector can identify
NUM_CLASSES = 7


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def objects_in_xml(xml_file):
    xml_list=[]
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for member in root.findall('object'):
        value = (member[0].text,
                 int(member[4][0].text),
                 int(member[4][1].text),
                 int(member[4][2].text),
                 int(member[4][3].text)
                 )
        xml_list.append(value)
    return xml_list
    
def imagelist(folder_path):
    exts = ['jpg', 'jpeg', 'bmp', 'png', 'gif']
    f = []
    for (dirpath, dirnames, filenames) in walk(folder_path):
        # if ext_name == None and isinstance(ext_name, str) and len(ext_name) > 0:
        for file in filenames:
            for ext in exts:
                if file.endswith(ext):
                    f.append(os.path.join(dirpath, file))
                    break
        # else:
        #     for file in filenames:
        #         f.append(os.path.join(dirpath, file))
        break
    return f;
# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
image_paths = imagelist(FOLDER_OF_IMAGES)
total_hit_rate = 0
for image_path in image_paths:
    print(image_path)
    image = cv2.imread(image_path)

    # rsize and crop background img
    x_ratio = IMAGE_MAX / image.shape[1];
    y_ratio = IMAGE_MAX / image.shape[0];
    min_ration = min(x_ratio, y_ratio);
    

    if (min_ration < 1.0):
        image = cv2.resize(image, (int(image.shape[1] * min_ration), int(image.shape[0] * min_ration)),
                        interpolation=cv2.INTER_CUBIC)

    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    
    # Draw the results of the detection (aka 'visulaize the results')

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.30)
    basepath, ext_name = os.path.splitext(image_path)
    xmlpath = basepath+".xml"

    objects = objects_in_xml(xmlpath)
    except_obj_num = len(objects)
#    print("boxes:")
#    print(boxes)
#    print("classes:")
#    print(classes[0])
#    print("scores:")
#    print(scores[0])

    hit_point = 0;
    for i in range(len(boxes[0])):
        
        score = scores[0][i]
        if score < 0.3:
            continue
#        print("score:")
#        print(score)
        box = boxes[0][i];
        guess_id = classes[0][i] # 1-base
        guess_id -= 1   #0 base
#        print(box)
#        print(box[1])
#        print(box[0])
#        print(box[3])
#        print(box[2])
        bb1 = {'x1':box[1]*image.shape[1],'y1':box[0]*image.shape[0],'x2':box[3]*image.shape[1],'y2':box[2]*image.shape[0]}
#        print(bb1)
        max_correspond = -1
        max_correspond_id = -1
        for j in range(len(objects)):
            object = objects[j];
            #print(object)
            bb2 = {'x1':object[1],'y1':object[2],'x2':object[3],'y2':object[4]}
            correspond = get_iou(bb1,bb2)
            #print("correspond:" + str(correspond))
            if correspond > max_correspond:
                max_correspond = correspond
                max_correspond_id = j;
        if max_correspond_id >= 0:
            object = objects[max_correspond_id]
            answer_name = object[0]
            if answer_name != categories[int(guess_id)]['name']:
                print("Expect:"+answer_name +", Actual:" + categories[int(guess_id)]['name'])
                continue
            del objects[max_correspond_id]
            i-=1
            if max_correspond > 0.5:
                hit_point += 1
            else:
                hit_point += max_correspond
            
        
    hit_rate=hit_point/except_obj_num
    print("Hit rate: {:.1%}".format(hit_rate))
    total_hit_rate+=hit_rate
    if not os.path.exists(VERIFY_RESULT_FOLDER):
        os.makedirs(VERIFY_RESULT_FOLDER)
    jpgpath = os.path.join(VERIFY_RESULT_FOLDER,  os.path.basename(image_path))
    
    cv2.imwrite(jpgpath, image)
    
if len(image_paths) > 0:
    total_hit_rate/=len(image_paths)
print("Total Hit rate: {:.1%}".format(total_hit_rate))
#    cv2.imshow('detector', image)
#    cv2.waitKey(0)

# Press any key to close the image
# cv2.waitKey(0)

# Clean up
#cv2.destroyAllWindows()
