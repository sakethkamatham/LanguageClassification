import tkinter as tk
import pickle
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from py3langid.langid import LanguageIdentifier, MODEL_FILE
import fasttext
from gensim.models import KeyedVectors
from spacy.lang.en.stop_words import STOP_WORDS
import string
import re
import nltk
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import stopwords
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.feature_selection import SequentialFeatureSelector
import re
import nltk
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import stopwords
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.feature_selection import SequentialFeatureSelector


LANGID='LangID'
FASTTXT = 'Fast Text'
SPACY = 'Spacy'
SVMGV = 'SVM - GloVe'
RFGV = 'Random Forest - GloVe'
RFEXTRACT = 'Random Forest - With Feature Extraction'
KNNGV = 'KNN - GloVe'
KNNEXTRACT = 'KNN - With Feature Extraction'
SVMEXTRACT = 'SVM - With Feature Extraction'



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
    if token not in punct and token not in stopwords1:
      clean_tokens.append(token)
  return clean_tokens

try:
    
    with open('/Users/saketh/Desktop/LanguageClassification/LanguageClassification/Models/langId_1_1.pkl', 'rb') as f1:
        langId = pickle.load(f1)
    fasttextModel = fasttext.load_model('lid.176.bin')
    with open('/Users/saketh/Desktop/LanguageClassification/LanguageClassification/Models/spacy.pkl', 'rb') as f3:
        spacyModel = pickle.load(f3)
    with open('/Users/saketh/Desktop/LanguageClassification/LanguageClassification/Models/SvmGv.pkl', 'rb') as f4:
        svmModelgv = pickle.load(f4)
    with open('/Users/saketh/Desktop/LanguageClassification/LanguageClassification/Models/RfGv.pkl', 'rb') as f5:
        RandomForestModelgv = pickle.load(f5)
    with open('/Users/saketh/Desktop/LanguageClassification/LanguageClassification/Models/KnnGv.pkl', 'rb') as f6:
        knnModelgv = pickle.load(f6)
    with open('/Users/saketh/Desktop/LanguageClassification/LanguageClassification/Models/KnnExtract.pkl', 'rb') as f7:
        knnModelex = pickle.load(f7)
    with open('/Users/saketh/Desktop/LanguageClassification/LanguageClassification/Models/SvmExtract.pkl', 'rb') as f8:
        svmModelex = pickle.load(f8)
    with open('/Users/saketh/Desktop/LanguageClassification/LanguageClassification/Models/RfExtract.pkl', 'rb') as f9:
        RandomForestModelex = pickle.load(f9)
except Exception as e:
    print(f'File Loading Exception is {e}')

glove_model = KeyedVectors.load_word2vec_format('./glove.6B.100d.txt', binary=False, no_header=True)




nlp = spacy.load("en_core_web_sm")
stopwords1 = list(STOP_WORDS)
punct = string.punctuation



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
    
def avgWordLength(x):
    avgWordLength = [w for w in re.split('\.|!|\?| ',x)]    #split a review into a list of words
    avgWordLength = list(filter(lambda x : x != '', avgWordLength))    #remove any empty elements in the list
    avgWordLength = np.mean([len(w) for w in avgWordLength])    #find the average of the lengths of all words in the review
    return avgWordLength

def avgSentLength(x):
    avgSentLength = [w for w in re.split('\.|!|\?',x)]    #split a review into a list of sentences
    avgSentLength = list(filter(lambda x : x != '', avgSentLength))    #remove any empty elements in the list
    avgSentLength = np.mean([len(w) for w in avgSentLength])    #find the average of the lengths of all words in the review
    return avgSentLength

def numWords(x):
    numWords = [w for w in re.split('\.|!|\?| ', x)]    #split a review into a list of words
    numWords = list(filter(lambda x : x != '', numWords))    #remove any empty elements in the list
    numWords = len(numWords)    #count the number of words in the review
    return numWords

def numVerbs(x):
    numVerbs = sum(1 for word, pos in pos_tag(word_tokenize(x)) if pos.startswith('VB') or pos.startswith('VBD') or pos.startswith('VBG') or pos.startswith('VBN')  or pos.startswith('VBP')  or pos.startswith('VBZ'))
    return numVerbs

def numAdj(x):
    numAdj = sum(1 for word, pos in pos_tag(word_tokenize(x)) if pos.startswith('JJ') or pos.startswith('JJR') or pos.startswith('JJS'))
    return numAdj

def numNou(x):
    numNou = sum(1 for word, pos in pos_tag(word_tokenize(x)) if pos.startswith('NN') or pos.startswith('NNP') or pos.startswith('NNS'))
    return numNou

def languagesToListP(x):
    lang = [s for s in re.split('\.|!|\?',x)]    #split a review into a list of sentences
    lang = list(filter(lambda x : x != '', lang))    #remove any empty elements in the list
    return lang

def checkForPassive(s):   
    doc = nlp(s)    #document the sentence based on spacy nlp
    entityTags = [word.dep_ for word in doc]    #get the tags for all words in the sentence
    passive = any(['nsubjpass' in tag for tag in entityTags])    #get whether any words are tagged as passive
    return passive

def checkOneReviewP(r):
  numPassive = 0
  for sentence in r:    #for each sentence in a review:
    passive = checkForPassive(str(sentence))    #use the checkForPassive method to get the number of words 
    if(passive):
      numPassive = numPassive + 1;    #if a sentence is flagged as passive, increment the number of passive sentences found for the review
  return numPassive

def languagesToListA(x):
    lang = [s for s in re.split('\.|!|\?',x)]    #split a review into a list of sentences
    lang = list(filter(lambda x : x != '', lang))    #remove any empty elements in the list
    return lang

def checkForActive(s):   
    doc = nlp(s)    #document the sentence based on spacy nlp
    entityTags = [word.dep_ for word in doc]    #get the tags for all words in the sentence
    active = any(['nsubj' in tag for tag in entityTags])    #get whether any words are tagged as active
    return active

def checkOneReviewA(r):
  numActive = 0
  for sentence in r:    #for each sentence in a review:
    active = checkForActive(str(sentence))    #use the checkForActive method to get the number of words 
    if(active):
      numActive = numActive + 1;    #if a sentence is flagged as active, increment the number of active sentences found for the review
  return numActive

def removePunctAndStopWords(x): 
  words = re.sub('[\.!\?]', '', x)    #remove periods, exclamation points, and question marks from the review
  words = [w for w in re.split(' ', words)]    #split a review into a list of words
  words = list(filter(lambda x : x != '', words))    #remove any empty elements in the list
  words = [w for w in words if w not in stopwords.words('english')]    #remove words that are stopwords based on nltk stopwords
  return words

def contentDiversity(x):
  n = len(x)    #n = number of tokens/words in a review
  v = 0         #v = number of unique tokens/words in a review
  uniqueWords = []    #list of unique words in a review
  for w in x:
    if w not in uniqueWords:    #if a word is not in the uniqueWords, add it to the list and increment v
      v = v + 1
      uniqueWords.append(w)
  try:    
    return v / n    #return v divided by n
  except ZeroDivisionError:
    return 0




# Create a dictionary to map language names to models
models = {LANGID: langId ,FASTTXT: fasttextModel,SVMGV: svmModelgv,SVMEXTRACT: svmModelex, RFGV:RandomForestModelgv,KNNGV:knnModelgv,KNNEXTRACT: knnModelex, SPACY : spacyModel,RFEXTRACT:RandomForestModelex}

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


    new_col=['AWL','ASL','NWO','NVB','NAJ','NNO','NPV','NAV','NST','TLN','CDV']
    language=df1.reindex(columns=[*df1.columns.tolist(), *new_col], fill_value=0)

    language['AWL']=language['Text'].apply(avgWordLength)
    language['ASL']=language['Text'].apply(avgSentLength)
    language['NWO'] = language['Text'].apply(numWords)
    language['NVB'] = language["Text"].apply(numVerbs)
    language['NAJ'] = language["Text"].apply(numAdj)
    language['NNO'] = language["Text"].apply(numNou)
    langList = language["Text"].apply(languagesToListP)

    finalNumPassive = []

    for entry in langList:
        result = checkOneReviewP(entry)    #for every review, use the checkOneReview method to get the number of passive sentences in the review
        finalNumPassive.append(result)    #store the results in finalNumPassive

    #store the results in the NPV column
    language['NPV'] = finalNumPassive

    langList = language["Text"].apply(languagesToListA)

    finalNumActive = []

    for entry in langList:
        result = checkOneReviewA(entry)    #for every review, use the checkOneReview method to get the number of active sentences in the review
        finalNumActive.append(result)    #store the results in finalNumActive

    #store the results in the NPV column
    language['NAV'] = finalNumActive
    language['NST'] = language['Text'].str.count('[\.!\?]')
    language['TLN'] = language['Text'].str.len()

    wordLists = language['Text'].apply(removePunctAndStopWords)
    language['CDV']  = wordLists.apply(contentDiversity)
    language = language.drop(['Text'], axis=1)


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
            elif model == SVMGV:
                prediction = svmModelgv.predict(g)
            elif model == RFGV:
                prediction = RandomForestModelgv.predict(g)
            elif model == SPACY:
                d = np.array([text])
                prediction = spacyModel.predict(d)
                print(f'Language is {prediction}')
            elif model == KNNGV:
                prediction = knnModelgv.predict(g)
            elif model == KNNEXTRACT:
                languageKnn = language.drop(['ASL','NPV','NST','TLN','CDV'], axis=1)
                prediction = knnModelex.predict(languageKnn)
            elif model == SVMEXTRACT:
                languageSvm = language[['AWL','NVB','NAJ']]
                prediction = svmModelex.predict(languageSvm)
            elif model == RFEXTRACT:
                languageRf = language[['AWL', 'NNO']]
                prediction = RandomForestModelex.predict(languageRf)
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
