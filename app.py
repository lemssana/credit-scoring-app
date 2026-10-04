import streamlit as st
import pandas as pd
import joblib

model = joblib.load('credit_scoring_model.pkl')

st.title('Кредитный скоринг заемщика')

person_income = st.number_input('Годовой доход ($)', value=50000)
loan_amnt = st.number_input('Сумма кредита ($)', value=10000)
loan_int_rate = st.number_input('Процентная ставка (%)', value=11.0)
person_age = st.number_input('Возраст', value=26)
person_emp_length = st.number_input('Стаж работы (лет)', value=3.0)

if st.button('Оценить риск'):
    input_df = pd.DataFrame(0, index=[0], columns=model.feature_names_in_)
    input_df['person_income'] = person_income
    input_df['loan_amnt'] = loan_amnt
    input_df['loan_int_rate'] = loan_int_rate
    input_df['person_age'] = person_age
    input_df['person_emp_length'] = person_emp_length
    input_df['loan_percent_income'] = loan_amnt / person_income if person_income > 0 else 0

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f'Высокий риск дефолта! Вероятность: {probability:.2%}')
    else:
        st.success(f'Низкий риск. Кредит одобрен. Вероятность дефолта: {probability:.2%}')