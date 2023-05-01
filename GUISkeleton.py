import tkinter as tk
import pickle
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from py3langid.langid import LanguageIdentifier, MODEL_FILE
import fasttext
from gensim.models import KeyedVectors



LANGID='LangID'
FASTTXT = 'Fast Text'
SPACY = 'Spacy'
SVM = 'SVM'
RF = 'Random Forest'
KNN = 'KNN'


nlp = spacy.load("en_core_web_sm")

from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS)

import string
punct = string.punctuation

def preprocess(sentence):
  doc = nlp(sentence)
  tokens = []
  for token in doc:
    if token.lemma_ != '-PRON-':
      temp = token.lemma_.lower().strip()
    else:
      temp = token.lower_
    tokens.append(temp)
  clean_tokens = []
  for token in tokens:
    if token not in punct and token not in stopwords:
      clean_tokens.append(token)
  return clean_tokens

# Load Pickle Files
try:
    
    with open('/Users/saketh/Desktop/LanguageClassification/LanguageClassification/Models/langId_1_1.pkl', 'rb') as f1:
        langId = pickle.load(f1)
    fasttextModel = fasttext.load_model('lid.176.bin')
    with open('/Users/saketh/Desktop/LanguageClassification/LanguageClassification/Models/spacy.pkl', 'rb') as f3:
        spacyModel = pickle.load(f3)
    with open('/Users/saketh/Desktop/LanguageClassification/LanguageClassification/Models/SvmGv.pkl', 'rb') as f4:
        svmModel = pickle.load(f4)
    with open('/Users/saketh/Desktop/LanguageClassification/LanguageClassification/Models/RfGv.pkl', 'rb') as f5:
        RandomForestModel = pickle.load(f5)
    with open('/Users/saketh/Desktop/LanguageClassification/LanguageClassification/Models/KnnGv.pkl', 'rb') as f6:
        knnModel = pickle.load(f6)
except Exception as e:
    print(f'File Loading Exception is {e}')

glove_model = KeyedVectors.load_word2vec_format('./glove.6B.100d.txt', binary=False, no_header=True)

def get_sentence_embedding(sentence):
    sentence_embedding = []
    for word in sentence.split():
        try:
            word_embedding = glove_model.get_vector(word)
            sentence_embedding.append(word_embedding)
        except KeyError:
            pass
    if sentence_embedding:
        return np.mean(sentence_embedding, axis=0)
    else:
        return np.zeros(100)



# Create a dictionary to map language names to models
models = {'LangID': langId ,'Fast Text': fasttextModel,'SVM': svmModel, 'Random Forest':RandomForestModel,'KNN':knnModel, 'Spacy' : spacyModel}

# Define the function to detect the language
def detect_language():
   # Get the text from the input box
    text = input_box.get('1.0', 'end-1c')
    # Remove any leading/trailing spaces
    text = text.strip()
    # If the text is not empty
    model = selected_model.get()

    a = {'Text': [text]}
    df1 = pd.DataFrame(a)
    g = [get_sentence_embedding(sentence) for sentence in df1['Text']]
    g = np.vstack(g)

    print(f'The Text Received is {text} AND the model selected is {model}')
    if text:
        try:
            if model == LANGID:
                d = {'Text': [text]}
                df = pd.DataFrame(d)
                langIdModel = LanguageIdentifier.from_pickled_model(langId, norm_probs=True)
                prediction = langIdModel.classify(df.iloc[0].Text)[0]
            elif model == FASTTXT:
                prediction = fasttextModel.predict(text)[0][0].split('__')[-1]
            elif model == SVM:
                prediction = svmModel.predict(g)
            elif model == RF:
                prediction = RandomForestModel.predict(g)
            elif model == SPACY:
                d = np.array([text])
                prediction = spacyModel.predict(d)
                print(f'Language is {prediction}')
            elif model == KNN:
                prediction = knnModel.predict(g)
            else:
                prediction = "Model Not found"

            print(f'Language is {prediction}')
         
            output_box.configure(state='normal')
            output_box.delete('1.0', 'end')
            output_box.insert('1.0', prediction_translator(prediction))
            output_box.configure(state='disabled')
        except Exception as e:
            # If an error occurred, update the output box with an error message
            output_box.configure(state='normal')
            output_box.delete('1.0', 'end')
            output_box.insert('1.0', 'Error: Could not detect language')
            output_box.configure(state='disabled')
            print(f'Exception is {e}')

placeholder1 = 'Text is in '
placeholder2 = '- Most Spoken Country/ies : '

def prediction_translator(prediction):
    print(f'Received Prediction Value is {type(prediction)}')
    if type(prediction) is not str:
        prediction = prediction[0]
        print(f'Precessed  is {prediction}')

    if(prediction =='en'):
        return (f'{placeholder1}English {placeholder2} Around the world')
    elif(prediction == 'ml'):
       return (f'{placeholder1}Malayalam {placeholder2} India')
    elif(prediction == 'hi'):
        return (f'{placeholder1}Hindi {placeholder2} India')
    elif(prediction == 'ta'):
        return (f'{placeholder1}Tamil {placeholder2} India')
    elif(prediction == 'pt'):
        return (f'{placeholder1}Portugese {placeholder2} Portugal, Brazil')
    elif(prediction ==  'fr'):
        return (f'{placeholder1}French {placeholder2} France')
    elif(prediction ==  'nl'):
        return (f'{placeholder1}Dutch {placeholder2} Netherlands')
    elif(prediction ==  'es'):
        return (f'{placeholder1}Spanish {placeholder2} Spain (España)')
    elif(prediction ==  'el'):
        return (f'{placeholder1}Greek {placeholder2} Greece')
    elif(prediction == 'ru'):
        return (f'{placeholder1}Russian {placeholder2} Russia')
    elif(prediction == 'da'):
        return (f'{placeholder1}Danish {placeholder2} Denmark')
    elif(prediction == 'it'):
        return (f'{placeholder1}Italian {placeholder2} Italy')
    elif(prediction ==  'tr'):
        return (f'{placeholder1}Turkish {placeholder2} Turkey')
    elif(prediction == 'sv'):
        return (f'{placeholder1}Swedish {placeholder2} Sweden')
    elif(prediction == 'ar'):
        return (f'{placeholder1}Arabic {placeholder2} The Middle East')
    elif(prediction == 'de'):
        return (f'{placeholder1}German {placeholder2} Germany')
    elif(prediction == 'kn'):
        return (f'{placeholder1}Kannada {placeholder2} India')
    else:
        return "Sorry ! It looks like this language was not trained by us."

            
# Create the GUI
root = tk.Tk()
root.title('Language Detection')

root.geometry('1000x1000')
root.eval('tk::PlaceWindow . center')

# Create the header and pack it
header_label = tk.Label(root, text='Language Classification Interface', font=('Arial', 45, 'bold'), pady=10)
header_label.pack(pady=40)

# Create the input box
input_box = tk.Text(root, height=5, width=50, font=('Helvetica', 30))
input_box.pack(padx=10, pady=10)

selected_model = tk.StringVar(root)
language_dropdown = tk.OptionMenu(root, selected_model, *models)
language_dropdown.pack()
language_dropdown.pack(pady=40)


# Create the button to detect the language
button = tk.Button(root, text='Detect Language', command=detect_language, bg='white', fg='Black')
button.pack(pady=40)

# Create the output box
output_box = tk.Text(root, height=5, width=50, state='disabled', font=('Helvetica', 30))
output_box.pack()


# Create the footer and pack it
footer_label = tk.Label(root, text='Kamatham | Webb | Kobayashi', font=('Helvetica', 35), pady=10)
footer_label.pack(side='bottom')


# Start the GUI
root.mainloop()
