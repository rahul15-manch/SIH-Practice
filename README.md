#  🖐️ **ASL Sign Language Recognition**

A deep learning-based project to recognize American Sign Language (ASL) letters from images.
Built with TensorFlow, OpenCV, and Streamlit, this app allows users to upload an image of a hand sign and get a real-time prediction.

## 🚀 **Features**

- 📸 Image Upload – Upload an image of a hand sign for prediction.

- 🧠 Deep Learning Model – Custom CNN trained on the ASL Alphabet dataset.

- ⚡ Preprocessing – Images are resized, normalized, and augmented for better accuracy.

- 🌐 Streamlit App – Interactive web app for deployment.

- 🔊 Text-to-Speech – Converts predicted output into speech using gTTS or pyttsx3.

## 📂 **Project Structure**
```
ASL-Sign-Language-Recognition/
│── app.py                 # Streamlit app
│── preprocessing.py       # Preprocessing utilities
│── src/
│   └── train.py           # Training script
│── models/
│   ├── sign_model.h5      # Trained CNN model
│   └── class_indices.json # Class label mapping
│── data/
│   └── raw/               # Dataset (ASL Alphabet dataset)
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
```

## ⚙️ **Installation**
```

# Clone the repository

git clone https://github.com/<your-username>/ASL-Sign-Language-Recognition.git
cd ASL-Sign-Language-Recognition


# Create virtual environment

python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows


# Install dependencies

pip install -r requirements.txt


# Run Streamlit app

streamlit run app.py
```

## 📊 **Dataset**

We use the ASL Alphabet Dataset
 containing 87,000+ images of hand signs representing 29 classes (A–Z + Space, Delete, Nothing).

## 🧠 **Model Training**

- Input size: 64x64 RGB images

- Architecture: Custom CNN with Conv2D, MaxPooling, Dense, Dropout layers

- Optimizer: Adam

- Loss: Categorical Crossentropy

- Accuracy: ~95% on validation set

To retrain the model:
```
python src/train.py
```
## 🌐 **Deployment on Streamlit Cloud**

- Push your code to GitHub.

- Add a requirements.txt file (dependencies).

- If needed, add runtime.txt with Python version (e.g., python-3.10.13).

- Deploy on Streamlit Cloud.

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow/Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-ffffff?style=for-the-badge&logo=plotly&logoColor=black)
![Seaborn](https://img.shields.io/badge/Seaborn-0099CC?style=for-the-badge&logoColor=white)
![gTTS](https://img.shields.io/badge/gTTS-FFDD00?style=for-the-badge&logo=google&logoColor=black)
![pyttsx3](https://img.shields.io/badge/pyttsx3-4B8BBE?style=for-the-badge&logo=python&logoColor=white)



## 👨‍💻 **Authors**

### 🌐 Connect with us  
## 👥 Team Members  

| Name | LinkedIn |
|------|----------|
| **Rahul Manchanda** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rahul-manchanda-3959b120a/) |
| **Tanishka** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/tanishka-mukhi09/) |
| **Kashish** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kashish-rana-6116691b5/) |
| **Mayank** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mayank-gaur-dev/) |



