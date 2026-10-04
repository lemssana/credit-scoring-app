# Credit Risk Scoring & ML Model

Проект по кредитному скорингу для оценки вероятности дефолта заемщика на основе финансовых и демографических данных.

## Стек технологий
* Python, Pandas, Scikit-Learn, Joblib
* Random Forest Classifier
* Streamlit (веб-интерфейс)

## Метрики качества
* ROC-AUC Score: 0.9339

## Ключевые факторы риска (Feature Importance)
1. Отношение суммы кредита к доходу (`loan_percent_income`)
2. Годовой доход заемщика (`person_income`)
3. Процентная ставка (`loan_int_rate`)

## Запуск проекта
1. Установи зависимости: `pip install pandas scikit-learn streamlit joblib`
2. Запусти веб-интерфейс: `streamlit run app.py`