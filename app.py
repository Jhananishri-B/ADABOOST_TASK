import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


@st.cache_data
def load_data():
    df = pd.read_csv(r"D:\AI WORKSHOP\TASK\BOOSTING_task\UCI_Credit_Card.csv")
    df = df.drop(columns=["ID"], errors="ignore")  # Drop ID if exists
    return df

df = load_data()

st.title("📊 Credit Card Default Prediction App")

st.write("### Dataset Preview")
st.dataframe(df.head())

X = df.drop(columns=["default.payment.next.month"])
y = df["default.payment.next.month"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)
model = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=200,
    learning_rate=0.5,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"### ✅ Model Accuracy: {acc:.2f}")


st.write("### 📈 Skewness of Numeric Features")
skew_vals = df.skew(numeric_only=True).sort_values()
st.write(skew_vals)

fig, ax = plt.subplots(figsize=(12,5))
sns.barplot(x=skew_vals.index, y=skew_vals.values, palette="coolwarm", ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.axhline(0, color="black", linestyle="--")
ax.set_title("Skewness of Features")
st.pyplot(fig)


st.write("## 🔮 Predict Default Payment")

with st.form("prediction_form"):
    limit_bal = st.number_input("Credit Limit (LIMIT_BAL)", min_value=0, max_value=1000000, value=20000)
    sex = st.selectbox("Sex (1 = Male, 2 = Female)", [1, 2])
    education = st.selectbox("Education (1=Grad School, 2=University, 3=High School, 4=Others)", [1, 2, 3, 4])
    marriage = st.selectbox("Marriage (1=Married, 2=Single, 3=Others)", [1, 2, 3])
    age = st.slider("Age", 18, 100, 30)

    pay_0 = st.selectbox("Last Month Payment Status (PAY_0)", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    bill_amt1 = st.number_input("Bill Amount (BILL_AMT1)", min_value=0, max_value=1000000, value=5000)
    pay_amt1 = st.number_input("Payment Amount (PAY_AMT1)", min_value=0, max_value=1000000, value=2000)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame([[
        limit_bal, sex, education, marriage, age, pay_0,
        bill_amt1, pay_amt1
    ]], columns=["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "BILL_AMT1", "PAY_AMT1"])

    
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[X.columns]

    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ Customer is **likely to DEFAULT** next month.")
    else:
        st.success("✅ Customer is **NOT likely to default** next month.")

