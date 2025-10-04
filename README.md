❤️ Full Heart Disease Prediction
Sprints x Microsoft – AI & Machine Learning Track (Final Project)








🧠 Project Overview

This project was developed as the final capstone of the Sprints x Microsoft Summer Camp (AI & Machine Learning Track).
It focuses on building an end-to-end Machine Learning pipeline to predict heart disease based on medical and lifestyle data.

The goal was to explore the full ML workflow — from data preprocessing and analysis to model deployment — demonstrating both technical and practical understanding of AI systems.

🎯 Objectives

Perform exploratory data analysis (EDA) to uncover patterns and relationships.

Build and compare multiple supervised learning models.

Apply dimensionality reduction and hyperparameter tuning to improve performance.

Deploy a Streamlit web app for real-time heart disease prediction.

🧩 Project Pipeline
1. Data Preprocessing

Handled missing values and data encoding (Label & One-Hot Encoding).

Standardized numerical features for balanced model performance.

Split dataset into training and testing sets.

2. Exploratory Data Analysis (EDA)

Visualized distributions, correlations, and outliers using Matplotlib and Seaborn.

Detected significant health indicators like cholesterol, age, and resting blood pressure.

3. Dimensionality Reduction

Applied Principal Component Analysis (PCA) to reduce feature space while retaining key information.

4. Model Building

Built and evaluated multiple ML models:

Logistic Regression

Random Forest Classifier

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Naive Bayes

5. Evaluation Metrics

Models were evaluated using:

Accuracy

Precision, Recall, and F1-Score

ROC & AUC Curves

Confusion Matrix

🏆 Best Model: Random Forest

Accuracy: 92%

AUC: 0.95

6. Unsupervised Learning

Implemented K-Means and Hierarchical Clustering for patient segmentation.

7. Deployment

Built a Streamlit web app for real-time prediction.

Integrated visualizations for user interaction and data insights.

Deployed using Ngrok for public access.

🧰 Tech Stack

Programming Language: Python

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, PCA

Visualization Tools: Matplotlib, Seaborn

Web Framework: Streamlit

Deployment: Ngrok

💻 How to Run Locally
# Clone this repository
git clone https://github.com/<your-username>/Full_Heart_Disease_Prediction_sprints_x_microsoft_Final_project.git

# Navigate into the project folder
cd Full_Heart_Disease_Prediction_sprints_x_microsoft_Final_project

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

📊 Results & Insights

Random Forest achieved the best overall accuracy and generalization performance.

PCA successfully reduced dimensionality while preserving interpretability.

Data visualization helped identify strong correlations between key variables such as age, cholesterol, and thalach (max heart rate).
