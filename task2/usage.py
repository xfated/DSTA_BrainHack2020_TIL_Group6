from extract_clothes import get_clothes_class


    
txt = input("input mission statement: ")
print('Mission statement', txt)

extractor = get_clothes_class('./nlp_model.hdf5','./tokenizer.pkl','./encoded_words.pkl')
desired_clothing = extractor.process_input(txt)
print(desired_clothing)