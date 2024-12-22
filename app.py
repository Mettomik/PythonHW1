import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Функция для получения текущей температуры через OpenWeatherMap API
def get_current_temperature(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric", "lang": "ru"}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data["main"]["temp"]
    elif response.status_code == 401:
        st.error("Ошибка: Неверный API-ключ.")
    else:
        st.error(f"Ошибка при получении данных: {response.status_code}")
    return None

# Определение текущего сезона
def get_current_season():
    current_date = datetime.now()
    if current_date.month in [12, 1, 2]:
        return 'winter'
    elif current_date.month in [3, 4, 5]:
        return 'spring'
    elif current_date.month in [6, 7, 8]:
        return 'summer'
    else:
        return 'autumn'

# Функция анализа данных
def analyze_city(city_data):
    rolling_mean = city_data['temperature'].rolling(window=30, min_periods=1).mean()
    rolling_std = city_data['temperature'].rolling(window=30, min_periods=1).std()

    seasonal_stats = city_data.groupby('season')['temperature'].agg(['mean', 'std']).reset_index()
    seasonal_profile = {
        row['season']: {"mean": row['mean'], "std": row['std']}
        for _, row in seasonal_stats.iterrows()
    }

    anomalies = city_data[np.abs(city_data['temperature'] - rolling_mean) > 2 * rolling_std]

    return rolling_mean, rolling_std, anomalies, seasonal_profile

# Функция для обучения модели Холта-Винтерса
def train_holt_winters_model(city_data):
    model = ExponentialSmoothing(
        city_data['temperature'],
        trend='add',
        seasonal='add',
        seasonal_periods=365
    )
    results = model.fit(optimized=True)
    return results

# Функция для прогноза с моделью Холта-Винтерса
def forecast_holt_winters(model, forecast_period):
    forecast = model.forecast(steps=forecast_period)
    return forecast

# Визуализация аномалий
def plot_temperature_with_anomalies(city_data, rolling_mean, rolling_std, anomalies):
    fig = go.Figure()

    # Линия температуры
    fig.add_trace(go.Scatter(
        x=city_data["timestamp"],
        y=city_data["temperature"],
        mode='lines',
        name='Температура',
        line=dict(color='blue'),
        opacity=0.7
    ))

    # Скользящее среднее
    fig.add_trace(go.Scatter(
        x=city_data["timestamp"],
        y=rolling_mean,
        mode='lines',
        name='Скользящее среднее (30 дней)',
        line=dict(color='orange', width=2)
    ))

    # Аномалии
    fig.add_trace(go.Scatter(
        x=anomalies["timestamp"],
        y=anomalies["temperature"],
        mode='markers',
        name='Аномалии',
        marker=dict(color='red', size=8, symbol='circle')
    ))

    fig.update_layout(
        title=f"Температурный профиль с аномалиями",
        xaxis_title="Дата",
        yaxis_title="Температура (°C)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig

# Основная функция Streamlit
def main():
    st.title("Анализ температуры и текущие данные OpenWeatherMap")

    # Загрузка файла
    uploaded_file = st.file_uploader("Загрузите CSV файл с историческими данными", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file, parse_dates=["timestamp"])
        st.write("Загруженные данные:")
        st.dataframe(data.head())

        # Выбор города
        cities = data["city"].unique()
        selected_city = st.selectbox("Выберите город", cities)

        # Обнуляем модель при смене города
        if "selected_city" in st.session_state and st.session_state.selected_city != selected_city:
            st.session_state.model = None
            del st.session_state["model"]


        st.session_state.selected_city = selected_city

        api_key = st.text_input("Введите API-ключ OpenWeatherMap")

        # Анализ данных для выбранного города
        city_data = data[data["city"] == selected_city].copy()
        city_data = city_data.sort_values(by="timestamp").reset_index(drop=True)

        rolling_mean, rolling_std, anomalies, seasonal_profile = analyze_city(city_data)

        # Текущая температура + описательная статистика по выбранному городу
        if api_key:
            current_temp = get_current_temperature(selected_city, api_key)
            if current_temp is not None:
                current_season = get_current_season()
                mean = seasonal_profile[current_season]["mean"]
                std = seasonal_profile[current_season]["std"]
                st.subheader("Текущая температура")
                st.write(f"{current_temp:.2f}°C")
                if abs(current_temp - mean) > 2 * std:
                    st.warning("Текущая температура является аномальной для сезона.")
                else:
                    st.success("Температура в норме для сезона.")

                st.subheader("Описательная статистика")
                st.write(f"Средняя температура: {city_data['temperature'].mean():.2f}°C")
                st.write(f"Стандартное отклонение: {city_data['temperature'].std():.2f}")
                st.write(f"Количество наблюдений: {len(city_data)}")
                st.write(f"Самая ранняя дата: {city_data['timestamp'].min().date()}")
                st.write(f"Самая поздняя дата: {city_data['timestamp'].max().date()}")

                # Вывод графика с аномалиями
                fig_anomalies = plot_temperature_with_anomalies(city_data, rolling_mean, rolling_std, anomalies)
                st.plotly_chart(fig_anomalies)

        # Кнопки для построения модели и прогноза
        # Если модель не обучена, предлагаем обучить
        if "model" not in st.session_state:
            if st.button("Обучить модель"):
                st.session_state.model = train_holt_winters_model(city_data)
                st.success("Модель успешно обучена!")

        # если модель обучена то предлагаем прогноз.
        if "model" in st.session_state:
            forecast_period = st.slider("Выберите период прогноза (дни)", 30, 730, 30)
            if st.button("Построить прогноз"):
                forecast = forecast_holt_winters(st.session_state.model, forecast_period)
                forecast_index = pd.date_range(start=city_data['timestamp'].iloc[-1], periods=forecast_period + 1, freq='D')[1:]
                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(x=forecast_index, y=forecast, mode='lines', name='Прогноз'))
                fig_forecast.update_layout(title="Прогноз температуры", xaxis_title="Дата", yaxis_title="Температура (°C)")
                st.plotly_chart(fig_forecast)

if __name__ == "__main__":
    main()
