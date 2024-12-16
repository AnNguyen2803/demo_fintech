import streamlit as st
import pandas as pd
from preprocessing import preprocess_data, prepare_user_input
from models import train_models, predict_credit_limit
from utils import load_data

# Title of the App
st.title("BNPL Credit Limit Prediction App")
st.write("""
Ứng dụng này sử dụng Machine Learning để:
1. Phân loại khách hàng có đủ điều kiện tín dụng hay không.
2. Dự đoán hạn mức tín dụng cho khách hàng đủ điều kiện.
""")

# Upload CSV
data_file_path = "data/customer_credit_data.csv"
data = load_data(data_file_path)

if data is not None:
    st.write("## Dữ liệu ban đầu:")
    st.dataframe(data.head())

    # Preprocess data
    data, label_encoders = preprocess_data(data)

    # Train models
    clf, reg, X_class, X_reg, metrics = train_models(data)

    # Display evaluation metrics
    st.write("## Kết quả mô hình:")
    st.write(f"**Độ chính xác mô hình phân loại:** {metrics['classification_accuracy']:.2f}")
    st.write(f"**Sai số trung bình của mô hình hồi quy (RMSE):** {metrics['regression_rmse']:.2f}")

    # User input for prediction
    st.write("## Dự đoán cho khách hàng mới:")
    user_input = prepare_user_input(st, label_encoders, X_class.columns)

    # Make predictions
    if user_input is not None:
        approved, credit_limit = predict_credit_limit(clf, reg, user_input)
        if approved:
            st.success("Khách hàng được duyệt tín dụng!")
            st.write(f"Hạn mức tín dụng dự đoán: **{credit_limit:,.0f} VND**")
        else:
            st.error("Khách hàng không đủ điều kiện tín dụng.")
