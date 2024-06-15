import streamlit as st
from PIL import Image
from tensorflow import *
from skimage.color import rgb2lab, lab2rgb
import numpy as np
from Model_python_file import *


def classify_image(model, input_image):
    # Convert image to grayscale
    pred = model.predict(input_image)
    output_class = class_names[np.argmax(pred)]
    return output_class


def main():
    st.title("Image Classification App")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Load the pre-trained model
        model_weights_path = "C:/Users/DELL/Desktop/Projects/ai-v-real/model/ai_imageclassifier.h5"
        # Load the model
        loaded_model = tf.keras.models.load_model(model_weights_path)

        # Display the uploaded image
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Original Image", use_column_width=True)

        # Colorize the image
        with st.spinner("Colorizing..."):
            tag = classify_image(loaded_model, input_image)

        # Display the colorized image
        st.write(tag, caption="Colorized Image", use_column_width=True)

if __name__ == "_main_":
    main()