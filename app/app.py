import os
import json
import requests
import streamlit as st
import pickle
import numpy as np
import pandas as pd

filename = 'model.pkl'
model = pickle.load(open(filename, 'rb'))

st.title('Welcome to Lending Club!')
st.header('Let\'s try to predict whether the loan would be accepted or rejected')

loan_amnt = st.number_input('Loan Amount')
title = st.text_input('Request Title', max_chars=30)
risk_score = st.number_input('Risk Score')
dti = st.number_input('Debt-To-Income Ratio')
addr_state = st.text_input('State (USA Only)', max_chars=2)
emp_length = st.selectbox('Employment Length', options=['10+ years', '3 years', '4 years', '6 years', '1 year', '7 years', '8 years', '5 years', '2 years', '9 years', '< 1 year'])

pred_button = st.button("Predict")

if pred_button:
    sample = {
        'loan_amnt': [loan_amnt], 
        'title': [title], 
        'risk_score': [risk_score], 
        'dti': [dti], 
        'addr_state': [addr_state],
        'emp_length': [emp_length]
    }

    prediction = model.predict(pd.DataFrame(sample))[0]

    st.write(f'Prediction: {"accepted" if prediction else "rejected"}')