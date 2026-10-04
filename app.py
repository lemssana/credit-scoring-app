import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

model = joblib.load('credit_scoring_model.pkl')

st.title("Кредитный скоринг заемщика")

st.write("Введите данные для оценки вероятности дефолта по кредиту:")

income = st.number_input("Годовой доход ($)", min_value=0, value=50000)
loan_amnt = st.number_input("Сумма кредита ($)", min_value=0, value=10000)
loan_int_rate = st.number_input("Процентная ставка (%)", min_value=0.0, value=11.0)
person_age = st.number_input("Возраст", min_value=18, max_value=100, value=26)
person_emp_length = st.number_input("Стаж работы (лет)", min_value=0.0, max_value=50.0, value=3.0)

if st.button("Оценить риск"):
    input_data = pd.DataFrame({
        'person_age': [person_age],
        'person_income': [income],
        'person_emp_length': [person_emp_length],
        'loan_amnt': [loan_amnt],
        'loan_int_rate': [loan_int_rate]
    })
    
    # Приводим к формату признаков модели (дополняем пропущенные колонки нулями)
    # Берем ожидаемые моделью фичи из самого RandomForest
    expected_features = model.feature_names_in_
    for col in expected_features:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[expected_features]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"Внимание: Высокий риск дефолта! Вероятность: {probability:.2%}")
    else:
        st.success(f"Низкий риск дефолта. Вероятность: {probability:.2%}")

    st.subheader("Объяснение решения модели (SHAP)")
    
   
    
    # Вычисление SHAP-значений для текущего клиента (берем срез для класса 1)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_data)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    # Передаем shap_values для класса 1
    shap.plots.waterfall(shap_values[0, :, 1], max_display=5, show=False)
    st.pyplot(fig)