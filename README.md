# 🧠 Brain Tumor Detection using CNN and VGG16

This project focuses on detecting **brain tumors from MRI scan images**
using **CNN** and **VGG16** algorithms. It combines image preprocessing, 
feature extraction, and a neural network model to classify MRI images 
as **tumor** or **non-tumor**.

📄 This work is also supported by a **published research paper**,
demonstrating both academic research and practical implementation.

------------------------------------------------------------------------

## 🖼️ Project Overview

The system performs the following steps:

1.  MRI brain scan images are collected\
2.  Images are preprocessed using Computer Vision techniques\
3.  Important features are extracted from images\
4.  A Deep Learning model is trained on labeled data\
5.  The model predicts whether a new MRI scan contains a tumor

This project demonstrates how **Machine Learning can assist in medical image
analysis**.

------------------------------------------------------------------------

## 🚀 Features

✔ MRI Image Preprocessing using OpenCV\
✔ Visualization of medical scan data\
✔ Tumor vs Non-Tumor classification\
✔ Deep Learning model built with TensorFlow/Keras\
✔ Model evaluation using accuracy and performance metrics\
✔ Research-backed implementation

------------------------------------------------------------------------

## 🛠 Tech Stack

-   **Python**\
-   **OpenCV** -- Image Processing\
-   **NumPy & Pandas** -- Data Handling\
-   **Matplotlib & Seaborn** -- Data Visualization\
-   **Scikit-learn** -- Model evaluation & splitting\
-   **TensorFlow / Keras** -- Deep Learning Model\
-   **Jupyter Notebook**

------------------------------------------------------------------------

## 🗂 Dataset Structure

The dataset used in this project is organized into **Training** and **Testing** folders.  
Each folder contains MRI brain scan images divided into four categories:

    dataset/
    │
    ├── training/
    │   └── glioma_tumor/
    │   └── meningioma_tumor/
    │   └── pituitary_tumor/
    │   └── no_tumor/
    │
    ├── testing/
        └── glioma_tumor/
        └── meningioma_tumor/
        └── pituitary_tumor/
        └── no_tumor/

------------------------------------------------------------------------

## 📌 Class Labels

| Folder Name        | Description |
|--------------------|-------------|
| `glioma_tumor`     | MRI images showing glioma tumors |
| `meningioma_tumor` | MRI images showing meningioma tumors |
| `pituitary_tumor`  | MRI images showing pituitary tumors |
| `no_tumor`         | Normal MRI brain scans without tumors |

This structure helps the model learn to distinguish between **different tumor types** as well as **healthy brain scans**.

------------------------------------------------------------------------

## 📂 Repository Structure

    Brain-Tumor-Detection/
    │
    ├── notebooks/
    │   └── brain-tumor-detection.ipynb
    │
    ├── dataset/
    │   └── training
    │   └── testing
    │
    ├── paper/
    │   └── Brain Tumor Detection Paper.pdf
    │
    ├── requirements.txt
    ├── README.md
    └── .gitignore

------------------------------------------------------------------------

## 📄 Research Publication

This project is based on our published research work on Brain Tumor
Detection using Machine Learning and Computer Vision.

**Title:** An Approach for Classification & Detection of Brain Tumor 
Using CNN & VGG-16\
**Year:** 2024

📥 Read the full paper here:\
`paper/Brain Tumor Detection Paper.pdf`

------------------------------------------------------------------------

## ⚙️ Installation

1️⃣ Clone the repository

    git clone https://github.com/riteshbhosale2071/BrainTumorDetectionProject.git

2️⃣ Move into the project folder

    cd Brain-Tumor-Detection

3️⃣ Install dependencies

    pip install -r requirements.txt

4️⃣ Launch Jupyter Notebook

    jupyter notebook

5️⃣ Open the notebook inside the `notebooks` folder and run all cells

------------------------------------------------------------------------

## 📊 Model Workflow

-   Image loading and resizing\
-   Noise removal and preprocessing\
-   Dataset labeling\
-   Train-test split\
-   Model training using CNN\
-   Performance evaluation

------------------------------------------------------------------------

## 📈 Future Improvements

🔹 Increase dataset size for better accuracy\
🔹 Try advanced CNN architectures\
🔹 Deploy as a web application\
🔹 Integrate real-time MRI scan prediction

------------------------------------------------------------------------

## 🤝 Contribution

Contributions, suggestions, and improvements are welcome!\
Feel free to fork this repository and submit a pull request.

------------------------------------------------------------------------

## ⭐ Support

If you found this project helpful, give it a ⭐ on GitHub!
