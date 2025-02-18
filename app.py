# Setup
import streamlit as st
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np

# Load Model
model = keras.models.load_model('model.keras')

# Create User Interface 

# title
with st.container():
    img_col, heading_col = st.columns([.5, 1.5])

    with img_col:
        st.image('Images/logo.png', width=150)

    with heading_col:
        st.write('# 😾:rainbow[IMAGE CLASSIFIER] 🐶')
        st.write('**:violet[This is a deep learning model built with a small version of the Xception neural net for classifying pet images into cat and dog.]**')

# method which classifies
def classifier(path):
    img = keras.utils.load_img(path, target_size=(180, 180))
    plt.imshow(img)

    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    dog_score = float(tf.sigmoid(predictions[0][0]))

    cat_score = 1 - dog_score

    if cat_score > dog_score:
        return st.write(f"# :rainbow[**CAT**]😾")
    else:
        return st.write(f"# :rainbow[**DOG**]🐶")
        
# Input and Output
with st.container():
    
    select_col, upload_col = st.columns([1,2])

    with select_col:
        selected_img = st.selectbox('**:blue[Select Image]** 🖼️', options=['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'dog1', 'dog2', 'dog3', 'dog4', 'dog5'])

        if selected_img:
            path = f'Images/demo images/{selected_img}.jpeg'
            st.image(path, width=230)
            classifier(path)

    with upload_col:
        uploaded_img = st.file_uploader('**:blue[Upload Image]** 🖼️', accept_multiple_files=False)   
    
        if uploaded_img:
            st.image(uploaded_img, width=230)
            classifier(uploaded_img)
    
# Container for sharing contents
with st.container():
     # five more cols for linking app with other platforms
    youtube_col, hfspace_col, madee_col, repo_col, linkedIn_col = st.columns([1,1.2,1.08,1,1], gap='small')

    # Youtube link
    with youtube_col:
        st.link_button('**VIDEO**', icon=':material/slideshow:', url='https://youtu.be/cJHsSOk7xqE', help='YOUTUBE')
    
    # Hugging Face Space link
    with hfspace_col:
        st.link_button('**HF SPACE**', icon=':material/sentiment_satisfied:', url='https://huggingface.co/spaces/madhav-pani/Cat_vs_Dog/tree/main', help='HUGGING FACE SPACE')

    # Madee Link
    with madee_col:
        st.button('**MADEE**', icon=':material/flight:', disabled=True, help='MADEE')

    # Repository Link
    with repo_col:
        st.link_button('**REPO**', icon=':material/code_blocks:', url='https://github.com/madhavpani/Cat_vs_Dog', help='GITHUB REPOSITORY')

    # LinkedIn link
    with linkedIn_col:
        st.link_button('**CONNECT**', icon=':material/connect_without_contact:', url='https://www.linkedin.com/in/madhavpani', help='LINKEDIN')
    