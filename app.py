import pandas
import streamlit as st
from AlternatingLeastSquares.AlternatingLeastSquaresRecommender import ALSRecommender

st.header("Рекомендательная система с использованием машинного обучения")

model = ALSRecommender("./AlternatingLeastSquares/Model")

user_for_prediction = st.selectbox("Выберите пользователя для рекомендации товаров", model.customers['customer_name'].values)

if st.button("Получить рекомедации"):
    recommendations = model.get_recommendations_with_names(user_for_prediction)
    result = ""

    dataframe_data = {"Рекомендации": recommendations[0], "Оценка Рекомендации": recommendations[1]}
    dataframe = pandas.DataFrame(dataframe_data)
    dataframe = dataframe[dataframe["Оценка Рекомендации"] > 0.00001]

    st.dataframe(dataframe, hide_index= True)
