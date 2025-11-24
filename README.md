# 🌱 Plant disease detector using CNNs
# AgroVision AI – Crop Disease Detection System

AgroVision AI is a simple, fast, and accessible crop disease detection system designed to help farmers identify plant diseases using just a smartphone. The system uses a Convolutional Neural Network (CNN) model and a Streamlit web application to provide instant diagnosis from leaf images.

## 🌾 Project Overview
Crop diseases cause massive losses in food production. Many farmers do not have access to experts who can diagnose diseases early. AgroVision AI solves this by allowing farmers to upload a leaf image and get instant feedback on whether the leaf is healthy or diseased.

This project was built for a hackathon and demonstrates how AI can support agriculture and food security

## 🚀 Features
- Upload a plant leaf image through the Streamlit app.
- AI model detects the type of disease.
- Fast results (within seconds).
- Simple and farmer-friendly interface.
- Deployed online using Streamlit Cloud.

## 📊 Dataset
- Total images: **329** locally collected plant leaf images.
- Crops: **Maize, Beans, Tomato, Rice, Guinea Corn**
- Images captured under controlled conditions.
- Preprocessing included resizing, normalization, and data augmentation.

## 🧠 Model
- Model Type: **Convolutional Neural Network (CNN)**
- Trained using TensorFlow/Keras
- Accuracy achieved: **94%**
- Optimized for small datasets using augmentation

## 🖥️ Streamlit App
The Streamlit app:
- Allows users to upload a leaf image
- Sends the image to the trained model
- Displays the predicted disease class

### How to Run Locally
```bash
pip install streamlit tensorflow pillow
streamlit run app.py
```

## ☁️ Deployment (Streamlit Cloud)
- The app was deployed using **Streamlit Cloud**
- Streamlit Cloud allowed hosting without setting up servers
- A shareable link was generated for easy access

## 📂 Project Structure
```
|-- app.py
|-- model/
|   |-- saved_model.h5
|-- dataset/
|-- README.md
```

## 🔮 Future Improvements
- Offline version for rural farmers
- Support for Hausa, Yoruba, and Igbo languages
- Larger dataset for more accuracy
- Integration with agricultural agencies
- Real-time disease confidence scoring


## 👥 Team
- Jane Nelson – Team Lead  
- Patience Mamman – IT Coordinator / Streamlit Developer  
- Yusuf Nimota – Project Manager  

---

## 📌 License
This project is for educational and hackathon purposes.

🚀 **Live Demo:** [Click here to view the app](https://peepearl-plant-disease-detector-model-app-mcgckn.streamlit.app/)
