from EP_api import findrobotIP, Robot
import cv2
import time
import threading
import argparse

import tensorflow as tf

import colorsys
import numpy as np
from timeit import default_timer as timer
from PIL import Image, ImageFont, ImageDraw

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


## VARIABLES
det_threshold = 0.7

# Load model function
def load_model(model_dir):
    model = tf.compat.v2.saved_model.load(str(model_dir),None)
    model = model.signatures['serving_default']
    return model

# Get model
saved_model_path = 'letter_recognition_model/saved_model'
detection_model = load_model(saved_model_path)

# Get labels for model
PATH_TO_LABELS = 'letter-detection.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


""" Takes an image array and runs through the model to generate predictions
"""
def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  output_dict = model(input_tensor)
 
  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  return output_dict

""" Gets inference from an image frame
Arguments
---------
model: tensorflow graph
    your object detection model
image_np: np.array
    Image in array format

Returns
-------
preds: list
    For predictions above detection threshold, list of top 5 predictions. in descending order of confidence
image_np: np.array
    Array of pixels for image. with bounding boxes included
"""
def show_inference(model, image_np):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
 
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)

    # For our own visualization
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)
    
    preds = []
    for i in range(len(output_dict['detection_boxes'])):
        preds.append([output_dict['detection_scores'][i], output_dict['detection_classes'][i], output_dict['detection_boxes'][i]])
    
    # Post processing  
    global det_threshold
    preds = [pred for pred in preds if pred[0] >= det_threshold] #filter by confidence threshold 
    preds.sort( key = lambda x: x[0], reverse = True) #sort in descending order
    preds = preds[:3]  # just take top 3 for bbox reference loc.

    
    return (preds, image_np)

"""
Arguments
---------
bbox: 2nd argument of prediction. e.g. pred[2]
    ymin, xmin, ymax, xmax (relative to image width and size)

Returns: tuple
--------
xmid: mid x of bbox
ymid: mid y of bbox
area: area of bbox. Can use to gauge relative distance of object. Large --> near af
    note that is in terms of relative size. adjust assumptions accordingly
"""
def get_center(bbox):
    ymin, xmin, ymax, xmax = bbox
    xmid = (xmax+xmin)/2
    ymid = (ymax+ymin)/2
    area = (ymax-ymin) * (xmax-xmin)
    return (xmid, ymid, area)


def total_bbox_area(preds):
    total_area = 0.0
    for pred in preds:
        total_area += get_center(pred[2])[2]
    return total_area


def close_stream(robot):
    print("Quitting...")
    robot.exit()


def main():

    #image = frame.copy() ## Or however you can your image from tello
    image = cv2.imread('map.jpg')
    preds, image_np = show_inference(detection_model, image)
    print(preds[0])
    print(preds[1])
    cv2.imshow("result", image_np)
    cv2.imwrite('predicted_map.jpg', image_np)



# def filter():
#     # image_np = cv2.imread('map.jpg')
#     image_np = [['@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@'], ['@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@'], ['@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@'], ['@', '@', '@', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '@', '@', '@', '@'], ['@', '@', '@', 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '@', '@', '@', '@'], ['@', '@', '@', 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '@', '@', '@', '@'], ['@', '@', '@', 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '@', '@', '@', '@'], ['@', '@', '@', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '@', '@', '@', '@'], ['@', '@', '@', 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, '@', '@', '@', '@'], ['@', '@', '@', 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, '@', '@', '@', '@'], ['@', '@', '@', 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, '@', '@', '@', '@'], ['@', '@', '@', 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, '@', '@', '@', '@'], ['@', '@', '@', 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, '@', '@', '@', '@'], ['@', '@', '@', 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, '@', '@', '@', '@'], ['@', '@', '@', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '@', '@', '@', '@'], ['@', '@', '@', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '@', '@', '@', '@'], ['@', '@', '@', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '@', '@', '@', '@'], ['@', '@', '@', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '@', '@', '@', '@'], ['@', '@', '@', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '@', '@', '@', '@'], ['@', '@', '@', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '@', '@', '@', '@'], ['@', '@', '@', 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, '@', '@', '@', '@'], ['@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@'], ['@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@'], ['@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@', '@']]
#     image_np = np.asarray(image_np)
#     #height = no. of rows
#     #width = no. of columns
#     height, width = image_np.shape
#     print(height)
#     print(width)
#     # f --> filter size == (f, f)
#     # s --> stride = how many steps to jump for next filter
#     f = 5
#     s = 1

#     ## details of your image 2d array:

#     num_steps_w = int((width - f)/s) + 1 
#     num_steps_h = int((height - f)/s) + 1

#     output = []

#     for h in range(num_steps_h):
#         h_start = h * s
#         h_end = h_start + f
        
#         for w in range(num_steps_w):
#             w_start = w * s
#             w_end = w_start + f
#             # print('{} {} {} {}'.format(h_start, h_end, w_start, w_end))
#             image_slice = image_np[h_start:h_end, w_start:w_end]        # 3 channel


#             sum = 0
#             for i in range(f):
#                 for j in range(f):
#                     if image_slice[i,j] == '1':
#                         sum += 1
#             if sum >= 20:
#                 # print(image_slice)
#                 output.append([(h_end+h_start)/2, (w_end+w_start)/2])

#     print(output)
# if __name__ == '__main__':
#     # main()
#     filter()

