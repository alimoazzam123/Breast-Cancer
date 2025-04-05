## 🔬 Breast Cancer Prediction Web App

Welcome to the **Breast Cancer Prediction** app – a powerful, user-friendly tool built with **Streamlit** and **Machine Learning** to predict whether a tumor is **Malignant** or **Benign** based on real diagnostic data.

![App Screenshot](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)
App link = https://breastcancerdetectap.streamlit.app
---

### 🚀 Features

- 🎛️ **Interactive UI** with easy numerical inputs
- 🧠 Predict using:
  - Logistic Regression
  - Random Forest Classifier
- ✅ Fast, accurate real-time results
- 📊 Built on the Breast Cancer Wisconsin (Diagnostic) dataset
- 📦 Packaged with `Streamlit` for deployment and sharing

---

### 🛠️ Tech Stack

- **Python**
- **Scikit-learn**
- **Streamlit**
- **Pandas / NumPy**
- **Matplotlib / Seaborn (optional for visuals)**

---

### 🖥️ How to Run Locally

1. **Clone this repository**  
```bash
git clone https://github.com/alimoazzam123/Breast-Cancer.git
cd breast-cancer-predictor
```

2. **Create a virtual environment** (optional but recommended)  
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install requirements**  
```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app**  
```bash
streamlit run cancer.py
```

---

### 📁 File Structure

```
├── cancer.py             # Main Streamlit app
├── model.pkl             # Trained ML models (logistic, random forest)
├── main.ipynb            # Jupyter notebook (model training, EDA, etc.)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

### 📊 Sample Input

Once the app is running, input the required values such as:
- Mean Radius
- Mean Texture
- Worst Area
- Smoothness Error  
...and many more.

🔍 The app will display a prediction:

> ✅ The Breast Cancer is **Benign**  
> 🚨 The Breast Cancer is **Malignant**

---

### 👨‍💻 Author

**Md Moazzam Ali**  
[LinkedIn](https://www.linkedin.com/in/alimoazzam82) | [Email](mailto:mdmoazzamali984@gmail.com)

---

### 📄 License

This project is licensed under the MIT License.

