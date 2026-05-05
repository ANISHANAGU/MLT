Machine Learning-Based PDS Demand Forecasting System (FCI)

📌Overview

This project builds a Machine Learning-based demand forecasting system for India’s Public Distribution System (PDS), managed by the Food Corporation of India (FCI).

It predicts demand for essential food grains like rice and wheat using historical data, helping improve planning, reduce wastage, and optimize distribution.

📌Problem Statement

The Public Distribution System faces:

Demand-supply imbalance
Overstocking and shortages
Lack of predictive planning tools

This project addresses these using Machine Learning.

🚀Solution Approach

Collected historical PDS data
Applied preprocessing techniques
Built regression-based ML models
Evaluated performance
Deployed using a dashboard

📌Dataset

Source: Open Government Data & NDAP
Includes:
Month-wise data
State-wise distribution
Rice & wheat distribution
Beneficiary count
Total food grain distribution

⚙️ Data Preprocessing

Data cleaning
Handling missing values
Feature selection
Normalization

🤖 Models Used

Linear Regression
Decision Tree Regressor
Random Forest Regressor

🏆 Best Model

Random Forest performed best due to handling non-linear patterns effectively.

📈 Evaluation Metrics

Mean Absolute Error (MAE)
Root Mean Square Error (RMSE)
R² Score

🔍 Key Insights

Seasonal demand patterns identified
State-wise variations observed
Improved accuracy using ensemble methods

🖥️ Web Application

Built using Streamlit to:

Visualize demand trends
Compare predictions vs actual values
Display model performance

🌐 Live App:

https://qmgixrt3dpwsta3vvxzgnx.streamlit.app/

<img width="1919" height="913" alt="image" src="https://github.com/user-attachments/assets/131383b0-095c-4397-aae7-d8b0d83c64ea" />
<img width="1919" height="924" alt="image" src="https://github.com/user-attachments/assets/1a89488e-8a4d-4ed7-a969-d4e590d623f5" />
<img width="1919" height="910" alt="image" src="https://github.com/user-attachments/assets/5efcbbbc-3079-4765-b155-5d6bc4fbbc72" />
<img width="1919" height="910" alt="image" src="https://github.com/user-attachments/assets/240ed2ab-d705-4949-86fc-ebc32ae3dd59" />

🛠️ Tech Stack

Python
Pandas
NumPy
Scikit-learn
TensorFlow
Streamlit

📂 Project Structure

MLT/
   -> app.py
   ->train_model.py
   -> model.pkl
   ->dataset.csv
   ->requirements.txt
   ->images/
   ->README.md
   
🚀 How to Run

git clone https://github.com/ANISHANAGU/MLT.git
cd MLT
pip install -r requirements.txt
streamlit run app.py

🎯 Impact

Reduces food grain wastage
Improves allocation efficiency
Supports data-driven decisions

⚠️ Limitations

Limited dataset
No real-time data
Deep learning not implemented

🔮 Future Work

Real-time data integration
Deep learning models
More food categories
Advanced recommendation system

👩‍💻 Team

Anishanagu B
Arunadevi R I
Deepshika S

🔗 Repository

https://github.com/ANISHANAGU/MLT
