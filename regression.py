import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle

model = tf.keras.models.load_model('regression_model.h5')

with open('reg_label_encorder_gender.pkl', 'rb') as file:
    label_encorder_gender = pickle.load(file)

with open('reg_onehot_encorder_geo.pkl', 'rb') as file:
    onehot_encorder_geo = pickle.load(file)

with open('reg_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title('Customer Salary Prediction')

geography = st.selectbox('Geography', onehot_encorder_geo.categories_[0])
gender = st.selectbox('Gender', label_encorder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
exited = st.selectbox('Exited', [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encorder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

geo_encoded = onehot_encorder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encorder_geo.get_feature_names_out(['Geography'])
)
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)
predicted_salary = prediction[0][0]

st.write(f'Predicted Salary: {predicted_salary:,.2f}')
