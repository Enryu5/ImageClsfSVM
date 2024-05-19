import os
import cv2
import pickle
import sklearn
import numpy as np
import streamlit as st
import preprocess as prs
from PIL import Image

# Fungsi untuk memuat model deteksi gambar
@st.cache_resource
def load_model():
    # Deserialize the model
    with open('new_model.pkl', 'rb') as f:
        model = pickle.load(f)

    return model

def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        save_dir = "img"  # Specify the directory where you want to save the files
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
        file_path = os.path.join(save_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None

# Fungsi untuk memprediksi gambar tangan
def predict_image(path, model):
    size = (150, 100)
    # Ubah gambar ke format yang sesuai
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)

    # Ektrak fitur gambar
    feature = prs.extract_features(image_resized)
    feature_array = np.array(feature)
    feature_to_predict = feature_array.reshape(1, -1)

    # Lakukan prediksi
    prediction = model.predict(feature_to_predict)
    return prediction

# Tampilkan antarmuka Streamlit
def main():
    st.title("Deteksi Gambar Tangan")
    st.write("Upload gambar tangan dan sistem akan memprediksi apakah itu gambar gunting, batu, atau kertas.")

    # Upload gambar
    uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Tampilkan gambar yang diunggah
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang diunggah', use_column_width=True)

        # Buat path gambar
        path = save_uploaded_file(uploaded_file)

        # Memuat model
        model = load_model()

        # Prediksi gambar
        prediction = predict_image(path, model)

        st.write(prediction)

    if "filename" in locals():
        os.remove(path)

if __name__ == "__main__":
    main()
