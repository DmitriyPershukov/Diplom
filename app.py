import pandas
import streamlit as st
from AlternatingLeastSquares.AlternatingLeastSquaresRecommender import ALSRecommender
from AlternatingLeastSquares.AlternatingLeastSquaresModel import AlternatingLeastSquares

st.header("Рекомендательная система с использованием машинного обучения")

model = ALSRecommender("./AlternatingLeastSquares/Model")

user_for_prediction = st.selectbox("Выберите пользователя для рекомендации товаров", model.customers['customer_name'].values)

if st.button("Получить рекомедации"):
    recommendations = model.get_recommendations_with_names(user_for_prediction)

    max_score = None
    for index, row in recommendations.iterrows():
        if max_score is None:
            max_score = row["Оценка Рекомендации"]
        recommendations.at[index, "Оценка Рекомендации"] = 100 / (max_score / row["Оценка Рекомендации"])

    st.dataframe(recommendations, hide_index= True)
