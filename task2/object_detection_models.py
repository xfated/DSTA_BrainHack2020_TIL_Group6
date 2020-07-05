from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


det_threshold = 0.6

categories = {
    '1':'tops',
    '2':'trousers',
    '3':'outerwear',
    '4':'dresses',
    '5':'skirts'
}

# Load model function
def load_model(model_dir):
    model = tf.compat.v2.saved_model.load(str(model_dir),None)
    model = model.signatures['serving_default']
    return model

# Get model
saved_model_path = 'faster_rcnn_resnet101_40k_with_dolls/saved_model'
detection_model = load_model(saved_model_path)

doll_model_path = 'doll_detector_25k_new/saved_model'
doll_model = load_model(doll_model_path)

# Get labels for model
PATH_TO_LABELS = 'object-detection.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

PATH_TO_LABELS_dolls = 'object-detection-doll.pbtxt'
category_index_dolls = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS_dolls, use_display_name=True)


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


def show_inference_dolls(model, image_np):
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
        category_index_dolls,
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


## NLP
class get_clothes_class():
    def __init__(self, model_path, tokenizer_path, encoded_word_dict_path):
        self.classes = ["outerwear", "tops", "trousers", "dresses", "skirts"]
        ### Import model
        
        self.model = tf.keras.models.load_model('nlp_model.hdf5')
            
        ### Import tokenizer
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
            f.close()
        ### Import decoding dictionary
        with open(encoded_word_dict_path, 'rb') as f:
            self.word_keys = pickle.load(f)
            f.close()
    
    '''plaintext to encoded words format'''
    def _encode_input(self, raw_string):
        raw_string_list = raw_string.split()
        encoded_string_list = [self.word_keys[i.lower()] for i in raw_string_list if i.lower() in self.word_keys.keys()]
        return ' '.join(encoded_string_list)    
    
    '''to process the encoded words for model prediction'''    
    def process_input(self, input_string):
        # Convert words into encoded input
        encoded_input_string = self._encode_input(input_string)
        print('encoded words: ', encoded_input_string)
        # Tokenise and transform your encoded input
        sequences = self.tokenizer.texts_to_sequences([encoded_input_string])
        processed_input = pad_sequences(sequences, maxlen=40, padding='post')
        # Pass the processed input into the prediction
        result = self.model.predict(processed_input)
        #Get classes result
        preds_labels = [[1 if x > 0.5 else 0 for idx,x in enumerate(i) ] for i in result]
        detected_classes = [self.classes[idx] for idx,item in enumerate(preds_labels[-1]) if item==1]
        return detected_classes

