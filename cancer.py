import streamlit as st
import pandas as pd
import numpy as np
import time
import pickle

# Load model
@st.cache_resource
def load_model():
    with open("model.pkl", 'rb') as file:
        models = pickle.load(file)
    return models

# Prediction function
def predict_data(data, model_name):
    models = load_model()
    if model_name == "logistic_regression":
        model = models["logistic_regression"]
    elif model_name == "Random_forest":
        model = models["Random_forest"]
    else:
        raise ValueError("Invalid model name")
    input_data = pd.DataFrame(data, index=[0])
    prediction = model.predict(input_data)
    return prediction

# Main Streamlit App
def main():
    st.set_page_config(page_title="🔬 Cancer Predictor", page_icon=":bar_chart:", layout="wide")
    st.title("🚀 **ML-Based Breast Cancer Prediction**")
    st.markdown("🔬 **Using Machine Learning to Estimate Cancer Type (Benign vs Malignant)**")

    # Sidebar layout
    st.sidebar.header("⚙️ **Configuration**")
    model_choice = st.sidebar.radio("Choose Model", ["🤖 Logistic Regression", "🧮 Random Forest"])

    # Input fields - replaced sliders with number inputs
    st.sidebar.markdown("### 🎯 Mean Cell Features")
    mean_radius = st.sidebar.number_input("📏 Mean Radius", value=14.0, min_value=5.0, max_value=30.0)
    mean_texture = st.sidebar.number_input("🎨 Mean Texture", value=20.0, min_value=10.0, max_value=40.0)
    mean_perimeter = st.sidebar.number_input("📐 Mean Perimeter", value=80.0, min_value=40.0, max_value=200.0)
    mean_area = st.sidebar.number_input("📦 Mean Area", value=500.0, min_value=100.0, max_value=2500.0)
    mean_smoothness = st.sidebar.number_input("🧽 Mean Smoothness", value=0.1, min_value=0.05, max_value=0.2)
    mean_compactness = st.sidebar.number_input("📦 Mean Compactness", value=0.1, min_value=0.01, max_value=1.0)
    mean_concavity = st.sidebar.number_input("🧲 Mean Concavity", value=0.2, min_value=0.0, max_value=1.5)
    mean_concave_points = st.sidebar.number_input("🕳️ Mean Concave Points", value=0.05, min_value=0.0, max_value=0.4)
    mean_symmetry = st.sidebar.number_input("🔁 Mean Symmetry", value=0.2, min_value=0.1, max_value=0.5)
    mean_fractal_dimension = st.sidebar.number_input("🧬 Mean Fractal Dimension", value=0.06, min_value=0.04, max_value=0.2)

    st.sidebar.markdown("### 🧪 Error in Features")
    radius_error = st.sidebar.number_input("📏 Radius Error", value=0.5, min_value=0.0, max_value=3.0)
    texture_error = st.sidebar.number_input("🎨 Texture Error", value=1.0, min_value=0.5, max_value=5.0)
    perimeter_error = st.sidebar.number_input("📐 Perimeter Error", value=5.0, min_value=1.0, max_value=25.0)
    area_error = st.sidebar.number_input("📦 Area Error", value=40.0, min_value=5.0, max_value=200.0)
    smoothness_error = st.sidebar.number_input("🧽 Smoothness Error", value=0.007, min_value=0.001, max_value=0.03)
    compactness_error = st.sidebar.number_input("📦 Compactness Error", value=0.02, min_value=0.01, max_value=0.4)
    concavity_error = st.sidebar.number_input("🧲 Concavity Error", value=0.03, min_value=0.01, max_value=0.4)
    concave_points_error = st.sidebar.number_input("🕳️ Concave Points Error", value=0.02, min_value=0.005, max_value=0.2)
    symmetry_error = st.sidebar.number_input("🔁 Symmetry Error", value=0.02, min_value=0.005, max_value=0.1)
    fractal_dimension_error = st.sidebar.number_input("🧬 Fractal Dimension Error", value=0.01, min_value=0.001, max_value=0.05)

    st.sidebar.markdown("### ⚠️ Worst Cell Features")
    worst_radius = st.sidebar.number_input("📏 Worst Radius", value=17.0, min_value=7.0, max_value=40.0)
    worst_texture = st.sidebar.number_input("🎨 Worst Texture", value=25.0, min_value=10.0, max_value=50.0)
    worst_perimeter = st.sidebar.number_input("📐 Worst Perimeter", value=100.0, min_value=50.0, max_value=250.0)
    worst_area = st.sidebar.number_input("📦 Worst Area", value=1000.0, min_value=150.0, max_value=4000.0)
    worst_smoothness = st.sidebar.number_input("🧽 Worst Smoothness", value=0.15, min_value=0.07, max_value=0.3)
    worst_compactness = st.sidebar.number_input("📦 Worst Compactness", value=0.3, min_value=0.02, max_value=1.5)
    worst_concavity = st.sidebar.number_input("🧲 Worst Concavity", value=0.4, min_value=0.0, max_value=1.5)
    worst_concave_points = st.sidebar.number_input("🕳️ Worst Concave Points", value=0.1, min_value=0.0, max_value=0.5)
    worst_symmetry = st.sidebar.number_input("🔁 Worst Symmetry", value=0.3, min_value=0.1, max_value=0.8)
    worst_fractal_dimension = st.sidebar.number_input("🧬 Worst Fractal Dimension", value=0.1, min_value=0.05, max_value=0.25)

    user_data = {
        "mean radius": mean_radius,
        "mean texture": mean_texture,
        "mean perimeter": mean_perimeter,
        "mean area": mean_area,
        "mean smoothness": mean_smoothness,
        "mean compactness": mean_compactness,
        "mean concavity": mean_concavity,
        "mean concave points": mean_concave_points,
        "mean symmetry": mean_symmetry,
        "mean fractal dimension": mean_fractal_dimension,
        "radius error": radius_error,
        "texture error": texture_error,
        "perimeter error": perimeter_error,
        "area error": area_error,
        "smoothness error": smoothness_error,
        "compactness error": compactness_error,
        "concavity error": concavity_error,
        "concave points error": concave_points_error,
        "symmetry error": symmetry_error,
        "fractal dimension error": fractal_dimension_error,
        "worst radius": worst_radius,
        "worst texture": worst_texture,
        "worst perimeter": worst_perimeter,
        "worst area": worst_area,
        "worst smoothness": worst_smoothness,
        "worst compactness": worst_compactness,
        "worst concavity": worst_concavity,
        "worst concave points": worst_concave_points,
        "worst symmetry": worst_symmetry,
        "worst fractal dimension": worst_fractal_dimension
    }

    if st.sidebar.button("📊 Predict"):
        with st.spinner("🕒 Processing your input... Please wait"):
            time.sleep(2)
            model_name = "logistic_regression" if model_choice == "🤖 Logistic Regression" else "Random_forest"
            prediction = predict_data(user_data, model_name)
            result = "(**Benign** 🟢)" if prediction[0] == 1 else "(**Malignant** 🔴)"
            st.success(f"Your prediction result is: {prediction[0]}\n{result}")
            user_data["Prediction"] = prediction[0]

        for key, value in user_data.items():
            if isinstance(value, np.ndarray):
                user_data[key] = value.tolist()

if __name__ == "__main__":
    main()


