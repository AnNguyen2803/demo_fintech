import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data):
    label_encoders = {}
    for column in ["Occupation", "Credit_History"]:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Generate synthetic columns
    data["Approved"] = pd.Series(data["Monthly_Income"]).apply(lambda x: 1 if x > 5000000 else 0)
    data["Credit_Limit"] = data["Monthly_Income"] * 1.5
    return data, label_encoders

def prepare_user_input(st, label_encoders, feature_columns):
    age = st.slider("Tuổi", 18, 70, 30)
    occupation = st.selectbox("Nghề nghiệp", label_encoders["Occupation"].classes_)
    monthly_income = st.number_input("Thu nhập hàng tháng", min_value=0, value=1000000, step=100000)
    credit_history = st.selectbox("Lịch sử tín dụng", label_encoders["Credit_History"].classes_)
    purchase_value = st.number_input("Tổng giá trị mua hàng 6 tháng gần nhất", min_value=0, value=10000000)
    purchase_freq = st.slider("Tần suất mua hàng", 0, 20, 5)
    on_time_payment = st.slider("Tỷ lệ thanh toán đúng hạn", 0.0, 1.0, 0.8)

    input_data = pd.DataFrame([[
        age,
        label_encoders["Occupation"].transform([occupation])[0],
        monthly_income,
        label_encoders["Credit_History"].transform([credit_history])[0],
        purchase_value,
        purchase_freq,
        on_time_payment
    ]], columns=feature_columns)
    return input_data
