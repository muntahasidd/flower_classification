🌸 Iris Flower Classifier — Streamlit App
Classify Iris flower species from an uploaded image using Machine Learning


🚀 Overview

This project is a Streamlit-based web app that classifies the species of an Iris flower using a machine learning model trained on the Iris dataset.

Because the Iris dataset contains 4 numerical measurements — sepal length, sepal width, petal length, petal width — and real images don’t include these values, I used creative feature extraction to convert an image into approximate numeric features.

This allows the ML model to make predictions based on any uploaded flower image


✨ Features

✔ Upload a flower image
✔ Image-based feature extraction (HSV & RGB analysis)
✔ Predict flower species:

Setosa

Versicolor

Virginica

✔ Shows confidence score (%)
✔ Displays probability distribution for all classes
✔ View extracted features


Streamlit

🧠 Model Details

Algorithm: Random Forest Classifier

Dataset: Iris dataset (UCI / Kaggle)

Accuracy: ~100% (dataset is small and very clean)

Model File: iris_model.pkl


📸 How Image Feature Extraction Works

Since Iris images do NOT contain actual sepal/petal measurements, the app:

Converts image → HSV

Computes average Hue, Saturation, Brightness

Computes RGB statistics

Maps these values into the Iris measurement range

This preserves the 4-feature requirement of the Iris model.




