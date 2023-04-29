import tkinter as tk
from tensorflow import keras

# # Load the pre-trained models
# langId = keras.models.load_model('path/to/your/model.h5')
# fasttextModel = keras.models.load_model('path/to/your/model.h5')
# spacyModel = keras.models.load_model('path/to/your/model.h5')
# svmModel = keras.models.load_model('path/to/your/model.h5')
# RandomForestModel = keras.models.load_model('path/to/your/model.h5')
# knnModel = keras.models.load_model('path/to/your/model.h5')

# # Create a dictionary to map language names to models
# models = {'langId': langId, 'fasttextModel': fasttextModel, 'Spanish': spacyModel, 'German': svmModel, 'RandomForestModel':RandomForestModel,'knnModel':knnModel}



# Doc.set_extension('language', default='UNK')

# Define the function to detect the language
def detect_language():
   # Get the text from the input box
    text = input_box.get('1.0', 'end-1c')
    # Remove any leading/trailing spaces
    text = text.strip()
    # If the text is not empty
    if text:
        try:
            # Detect the language
            language = detect(text)
            # Update the output box with the detected language
            output_box.configure(state='normal')
            output_box.delete('1.0', 'end')
            output_box.insert('1.0', language)
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

selected_model = tk.StringVar(root)
selected_model.set(models[0]) 

model_dropdown = tk.OptionMenu(root, selected_model, *models[0])
model_dropdown.pack()

root.geometry('1000x1000')
root.eval('tk::PlaceWindow . center')

# Create the header and pack it
header_label = tk.Label(root, text='Language Detection Interface', font=('Arial', 45, 'bold'), pady=10)
header_label.pack()

# Create the input box
input_box = tk.Text(root, height=5, width=50, font=('Helvetica', 14))
input_box.pack(padx=10, pady=10)


# Create the button to detect the language
button = tk.Button(root, text='Detect Language', command=detect_language, bg='white', fg='Black')
button.pack(pady=10)

# Create the output box
output_box = tk.Text(root, height=5, width=50, state='disabled', font=('Helvetica', 30))
output_box.pack()


# Create the footer and pack it
footer_label = tk.Label(root, text='Kamatham | Webb | Kobayashi', font=('Helvetica', 35), pady=10)
footer_label.pack(side='bottom')


# Start the GUI
root.mainloop()
