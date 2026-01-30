import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Загрузка модели
model = joblib.load("airline_satisfaction_model.pkl")

st.set_page_config(
    page_title="Airline Satisfaction Predictor",
    page_icon="✈️",
)

st.title("✈️ Прогноз удовлетворённости клиента")
st.write("Введите данные клиента и получите прогноз.")

st.sidebar.header("Параметры клиента")

# --- Ввод данных ---
age = st.sidebar.slider("Возраст", 18, 80, 35)
flight_distance = st.sidebar.number_input("Дистанция полёта", 100, 10000, 1500)

departure_delay = st.sidebar.number_input("Задержка вылета (мин)", 0, 500, 0)
arrival_delay = st.sidebar.number_input("Задержка прибытия (мин)", 0, 500, 0)

gender = st.sidebar.selectbox("Пол", ["Male", "Female"])
customer_type = st.sidebar.selectbox("Тип клиента", ["Loyal Customer", "disloyal Customer"])
travel_type = st.sidebar.selectbox("Тип поездки", ["Business travel", "Personal Travel"])
flight_class = st.sidebar.selectbox("Класс", ["Eco", "Eco Plus", "Business"])

# сервисные оценки
st.sidebar.subheader("Оценки сервиса (0–5)")

service_cols = [
    'Inflight wifi service',
    'Departure/Arrival time convenient',
    'Ease of Online booking',
    'Gate location',
    'Food and drink',
    'Online boarding',
    'Seat comfort',
    'Inflight entertainment',
    'On-board service',
    'Leg room service',
    'Baggage handling',
    'Checkin service',
    'Inflight service',
    'Cleanliness'
]

service_features = {}
for feature in service_cols:
    service_features[feature] = st.sidebar.slider(feature, 0, 5, 3)

# --- Формируем DataFrame ---
data = {
    "Age": age,
    "Flight Distance": flight_distance,
    "Departure Delay in Minutes": departure_delay,
    "Arrival Delay in Minutes": arrival_delay,
    "Gender": gender,
    "Customer Type": customer_type,
    "Type of Travel": travel_type,
    "Class": flight_class,
    **service_features
}

df = pd.DataFrame([data])

# --- ПРЕДОБРАБОТКА ДАННЫХ (такая же, как при обучении) ---

# 1. Преобразование категориальных признаков в числовые
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Customer Type'] = df['Customer Type'].map({'Loyal Customer': 1, 'disloyal Customer': 0})
df['Type of Travel'] = df['Type of Travel'].map({'Business travel': 1, 'Personal Travel': 0})

# 2. One-hot encoding для класса
df = pd.get_dummies(df, columns=['Class'], dtype=int)

# 3. Убедимся, что все нужные колонки присутствуют
# Создаем список всех ожидаемых колонок
expected_class_columns = ['Class_Eco', 'Class_Eco Plus', 'Class_Business']
for col in expected_class_columns:
    if col not in df.columns:
        df[col] = 0

# 4. Упорядочиваем колонки в том же порядке, что и при обучении
# Получаем список колонок из препроцессора модели
preprocessor = model.named_steps['prep']

# Для числовых колонок
num_cols = preprocessor.transformers_[0][2]

# Для категориальных колонок
cat_cols = preprocessor.transformers_[1][2]

# Все колонки в правильном порядке
all_cols = num_cols + cat_cols

# Реиндексируем DataFrame, добавляя отсутствующие колонки с нулями
df = df.reindex(columns=all_cols, fill_value=0)

# --- Предсказание ---
if st.button("Сделать прогноз"):
    try:
        proba = model.predict_proba(df)[0][1]

        st.subheader("Результат")
        st.metric(
            label="Вероятность удовлетворённости",
            value=f"{proba:.2%}"
        )

        if proba > 0.5:
            st.success("Клиент скорее всего удовлетворён 🙂")
        else:
            st.error("Клиент скорее всего не удовлетворён 😕")

    except Exception as e:
        st.error(f"Ошибка при предсказании: {str(e)}")
        st.write("Проверьте, что все колонки присутствуют:")
        st.write(f"Колонки в данных: {list(df.columns)}")
        st.write(f"Всего колонок: {len(df.columns)}")