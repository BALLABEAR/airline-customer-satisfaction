import streamlit as st
import pandas as pd
import joblib

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
service_features = {}
for feature in [
    "Inflight wifi service", "Seat comfort", "Inflight entertainment",
    "On-board service", "Cleanliness"
]:
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

# --- Предсказание ---
if st.button("Сделать прогноз"):
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
