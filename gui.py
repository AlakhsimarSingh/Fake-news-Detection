from tkinter import *
from tkinter import ttk
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import pickle as pkl

portStem = PorterStemmer()
m = pkl.load(open('Model.pkl','rb'))
with open("Vectorizer.pkl", "rb") as file:
    vectorizer = pkl.load(file)

def stemming(content):
    stemed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemed_content = stemed_content.lower()
    stemed_content = stemed_content.split()
    stemed_content = [portStem.stem(word) for word in stemed_content if not word in stopwords.words('english')]
    stemed_content = ' '.join(stemed_content)
    return stemed_content
def on_entry_click(event, entry_widget, placeholder):
    if entry_widget.get() == placeholder:
        entry_widget.delete(0, "end")
        entry_widget.config(foreground='black')

def on_focus_out(event, entry_widget, placeholder):
    if entry_widget.get() == "":
        entry_widget.insert(0, placeholder)
        entry_widget.config(foreground='#A2A2A2')

def create_entry_with_placeholder(parent, placeholder_text):
    e = ttk.Entry(parent, font=('Century', 12), width=20)
    e.insert(0, placeholder_text)
    e.config(foreground='#A2A2A2', width=40)  
    e.bind("<FocusIn>", lambda event, entry_widget=e, placeholder=placeholder_text: on_entry_click(event, entry_widget, placeholder))
    e.bind("<FocusOut>", lambda event, entry_widget=e, placeholder=placeholder_text: on_focus_out(event, entry_widget, placeholder))
    e.pack(padx=10, pady=10)
    return e

def submit():
    author = e1.get()
    title = e2.get()
    content = e3.get()
    # Prepare input data for prediction
    input_data = pd.DataFrame({'author': [author], 'title': [title], 'text': [content]})
    # Perform prediction
    # result = prediction(input_data)
    # Display prediction result
    input_data['title'] = input_data['title'].apply(stemming)
    input_dataset = input_data['title'].values
    input_dataset = vectorizer.transform(input_dataset)
    inp = np.asarray(input_dataset)
    inp.reshape(1, -1)
    result = m.predict(input_dataset)
    result_label.config(text=result[0], font=('Century', 14), wraplength=500)

root = Tk()
root.geometry("600x700")
root.title("Fake News Detection")

# Set the background image
bg = PhotoImage(file="F:\\ML Projects\\Fake news prediction\\newspaper-502778 1.png") 
label1 = Label(root, image=bg) 
label1.place(x=0, y=0, relwidth=1, relheight=1)
label_title = Label(root, text="Fake News Detection", font=('Century', 20))
label_title.pack(pady=20, padx=10)

# Create a frame for input fields
input_frame = Frame(root)
input_frame.pack(pady=10)

# Create the entry widgets
e1 = create_entry_with_placeholder(input_frame, "Enter the Author")
e2 = create_entry_with_placeholder(input_frame, "Enter the Title of the Article")
e3 = create_entry_with_placeholder(input_frame, "Enter the Content of the Article")

# Create a submission button
submit_button = ttk.Button(root, text="Submit", command=submit)
submit_button.pack(pady=10)

# Create a label to display prediction result
result_label = Label(root, text="", font=('Century', 14))
result_label.pack(pady=10)

root.mainloop()
