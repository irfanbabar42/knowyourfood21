import tensorflow as tf
model = tf.keras.models.load_model('model_inception.hdf5')
import streamlit as st
st.write("""
         Food Classification 21
         """
         )
st.write("Image classification web app to predict the type food made on streamlit")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])



from PIL import Image, ImageOps
import numpy as np




classes = ["apple_pie",
            "baby_back_ribs",
            "baklava",
            "beef_carpaccio",
            "beef_tartare",
            "beet_salad",
            "beignets",
            "bibimbap",
            "bread_pudding",
            "breakfast_burrito",
            "bruschetta",
            "caesar_salad",
            "cannoli",
            "caprese_salad",
            "carrot_cake",
            "ceviche",
            "cheesecake",
            "cheese_plate",
            "chicken_curry",
            "chicken_quesadilla",'other']


def import_and_predict(image_data, model):
    
        size = (299,299)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img_resize = image/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
        
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file).resize((200,200),Image.LANCZOS)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    prediction = np.argmax(prediction)
    st.write("Predicted Class: "+ classes[prediction])
    