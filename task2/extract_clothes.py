import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
    
    
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


### Example usage ###
# clothes=get_clothes_class('./fashion/model.pkl','./fashion/cvect.pkl','./fashion/encoded_words.pkl')
# result = clothes.process_input("blouse")
# print(result)