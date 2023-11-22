# Import necessary libraries
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import streamlit as st
import speech_recognition as sr
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker

# Calling spellcheck function
spell = SpellChecker()

# Load the model and tokenizer
model = load_model('next_word.h5')
tokenizer = pickle.load(open('token.pkl', 'rb'))

# Define function to predict next words
def predict_next_words(model, tokenizer, text):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = np.array(sequence)
    preds = np.argmax(model.predict(sequence))
    predicted_word = ""

    for key, value in tokenizer.word_index.items():
        if value == preds:
            predicted_word = key
            break
    return predicted_word

# Streamlit app
def main():
    st.set_page_config(
        page_title="Advancing Typing Assistant",
        page_icon="📚",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # Set background image
    st.markdown(
        """
        <style>
            body {
                background-image: url('background.jpg');
                background-size: cover;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Advancing Typing Assistant For Elderly Users")

    # Sidebar with user input
    n = st.sidebar.radio("Choose an option:", ["Text", "Speech"])
    if n == "Text":
        text = st.text_input("Enter your text:")
    elif n == "Speech":
        recognizer = sr.Recognizer()
        with sr.Microphone() as mic:
            st.warning("Please speak for speech recognition...")
            audio = recognizer.listen(mic)
            text = recognizer.recognize_google(audio)

    # Button to trigger the prediction
    if st.button("Predict Next Word"):
        st.subheader("Original Input:")
        st.write(text)

        # Spell checking and prediction
        words = word_tokenize(text.lower())
        corrected_words = [spell.correction(word) for word in words]
        corrected_text = " ".join(corrected_words)

        st.subheader("Spell-Corrected Input:")
        st.write(corrected_text)

        try:
            # Get the last 3 words for prediction
            last_three_words = corrected_words[-3:]
            st.subheader("Last Three Words for Prediction:")
            st.write(last_three_words)

            # Predict next word
            predicted_word = predict_next_words(model, tokenizer, last_three_words)

            # Display predicted word in a colored box
            st.subheader("Predicted Next Word:")
            st.markdown(
                f'<div style="background-color:#ddd;padding:10px;border-radius:10px;">{predicted_word}</div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
