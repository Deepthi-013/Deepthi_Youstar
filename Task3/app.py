import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship', 'truck']

model = tf.keras.models.load_model("cifar10_model.keras")

st.title("CIFAR-10 Image Classifier")

img_file = st.file_uploader("Upload an image (32x32)", type=['jpg', 'png'])

if img_file is not None:
    original_img = Image.open(img_file)
    st.image(original_img, caption='Original Image', use_column_width=True)

    img = original_img.convert('RGB').resize((32, 32))
    img_array = np.array(img).astype("float32") / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0]

    top3_idx = pred.argsort()[-3:][::-1]

    st.markdown("**Top 3 Predictions:**")
    for i in top3_idx:
        st.write(f"{labels[i]}: {pred[i]*100:.2f}%")
