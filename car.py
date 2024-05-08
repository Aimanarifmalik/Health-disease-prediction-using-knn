import streamlit as st
import numpy as np
import pandas as pd
import pickle

with open('scaler_model.pkl', 'rb') as f:
        scaler = pickle.load(f)
with open('model.pkl','rb')   as f:
        model=pickle.load(f)




st.title("Heart Disease Prediction")
st.write("Please enter the following details to predict whether you have heart disease or not:")

age =  int(st.number_input("Age:"))
sex = st.radio("Sex:", ['female', 'male'])

if sex == 'female':
  sex=0
else :
  sex =1
cp = int(st.selectbox("Chest Pain Type:", [0, 1, 2, 3]))
trestbps =int( st.number_input("Resting Blood Pressure:", min_value=0, max_value=300, step=1))
chol = int(st.number_input("Cholesterol (mg/dl):", min_value=0, max_value=600, step=1))
fbs = int(st.selectbox("Fasting Blood Sugar > 120 mg/dl:", [0, 1]))
restecg = int(st.selectbox("Resting Electrocardiographic Results:", [0, 1, 2]))
thalach = int(st.number_input("Maximum Heart Rate Achieved:", min_value=0, max_value=300, step=1))
exang = int(st.selectbox("Exercise Induced Angina:", [0, 1]))
oldpeak = float(st.number_input("ST Depression Induced by Exercise Relative to Rest:", min_value=0.0, max_value=10.0, step=0.1))
slope = int(st.selectbox("Slope of the Peak Exercise ST Segment:", [0, 1, 2]))
ca = int(st.selectbox("Number of Major Vessels Colored by Flouroscopy:", [0, 1, 2, 3]))
thal = int(st.selectbox("Thalassemia:", [0, 1, 2, 3]))
    
data=[[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
       exang, oldpeak, slope, ca, thal]]
columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal']
  
df=pd.DataFrame(data, columns=columns)    
X_scaled=scaler.transform(df)

prediction=model.predict(X_scaled)

if st.button('Submit'):
        st.write("Prediction:", "Heart Disease Present" if prediction[0] == 1 else "No Heart Disease")

