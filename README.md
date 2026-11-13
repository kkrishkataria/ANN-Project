# Customer Churn Prediction

A Streamlit web app that uses an artificial neural network (ANN) to estimate whether a bank customer is likely to churn.

## What is included

- `app.py` — Streamlit interface for entering customer details and viewing the churn probability.
- `experiments.ipynb` — model training and experimentation notebook.
- `model.h5` — trained TensorFlow/Keras model.
- `scaler.pkl`, `label_encorder_gender.pkl`, `onehot_encorder_geo.pkl` — saved preprocessing objects used by the app.
- `Churn_Modelling.csv` — source dataset.

## Run locally

1. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   streamlit run app.py
   ```

4. Open the local URL displayed by Streamlit, enter a customer's information, and review the predicted churn probability.

## Inputs

The app accepts customer geography, gender, age, balance, credit score, salary, tenure, product count, credit-card status, and activity status. It applies the saved encoders and scaler before passing the data to the trained ANN.

## Prediction rule

The model returns a probability from 0 to 1. The app labels customers with a probability above `0.50` as likely to churn.

