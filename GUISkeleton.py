import tkinter as tk
from tensorflow import keras
import pickle

# Load th5
# langId = keras.models.load_model('path/to/your/model.h5')
# fasttextModel = keras.models.load_model('path/to/your/model.h5')
# spacyModel = keras.models.load_model('path/to/your/model.h5')
# svmModel = keras.models.load_model('path/to/your/model.h5')
# RandomForestModel = keras.models.load_model('path/to/your/model.h5')
# knnModel = keras.models.load_model('path/to/your/model.h5')

#Load Pickle Files
with open('models/model.pkl', 'rb') as f1:
    langId = pickle.load(f1)
with open('models/model.pkl', 'rb') as f2:
    fasttextModel = pickle.load(f2)
with open('models/model.pkl', 'rb') as f3:
    spacyModel = pickle.load(f3)
with open('models/model.pkl', 'rb') as f4:
    svmModel = pickle.load(f4)
with open('models/model.pkl', 'rb') as f5:
    RandomForestModel = pickle.load(f5)
with open('models/model.pkl', 'rb') as f6:
    knnModel = pickle.load(f6)

# Create a dictionary to map language names to models
models = {'Lang ID': langId, 'Fast Text': fasttextModel, 'Spacy': spacyModel, 'SVM': svmModel, 'Random Forest':RandomForestModel,'KNN':knnModel}

# Define the function to detect the language
def detect_language():
   # Get the text from the input box
    text = input_box.get('1.0', 'end-1c')
    # Remove any leading/trailing spaces
    text = text.strip()
    # If the text is not empty
    model = selected_model.get()
    print(f'Received Text is : {text} and the Chosen Model is : {model}')
    if text:
        try:
            if selected_model == 'Lang ID':
                prediction = langId.predict(text)
            elif selected_model == 'Fast Text':
                prediction = fasttextModel.predict(text)
            elif selected_model == 'Spacy':
                prediction = spacyModel.predict(text)
            elif selected_model == 'SVM':
                prediction = svmModel.predict(text)
            elif selected_model == 'Random Forest':
                prediction = RandomForestModel.predict(text)
            elif selected_model == 'KNN':
                prediction = knnModel.predict(text)
            else:
                prediction = 'Model Not Found'

            output_box.configure(state='normal')
            output_box.delete('1.0', 'end')
            output_box.insert('1.0', prediction)
            output_box.configure(state='disabled')
        except:
            # If an error occurred, update the output box with an error message
            output_box.configure(state='normal')
            output_box.delete('1.0', 'end')
            output_box.insert('1.0', 'Error: Could not detect language')
            output_box.configure(state='disabled')

            
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
