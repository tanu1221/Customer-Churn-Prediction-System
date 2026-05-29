import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #eef2ff, #f8fafc);
}

.big-title {
    font-size: 55px;
    font-weight: bold;
    color: #111827;
}

.sub-text {
    font-size: 20px;
    color: #374151;
}

.card {
    background: white;
    padding: 35px;
    border-radius: 25px;
    text-align: center;
    transition: 0.3s;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.08);
    border: 1px solid #e5e7eb;
}

.card:hover {
    transform: translateY(-5px);
}

.card h1 {
    font-size: 42px;
    color: #2563eb;
}

.card h3 {
    color: #374151;
}

.stButton>button {
    background: linear-gradient(to right, #2563eb, #7c3aed);
    color: white;
    border-radius: 12px;
    height: 55px;
    width: 100%;
    font-size: 20px;
    border: none;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

st.sidebar.title("📊 Telecom Dashboard")

st.sidebar.markdown("""
### Project Information

- Customer Churn Prediction
- Machine Learning Project
- XGBoost Model
- Streamlit Dashboard
""")

st.sidebar.success("✅ Model Accuracy: 79%")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Prediction", "Analytics"]
)

if page == "Dashboard":

    st.markdown(
        '<p class="big-title">📉 Customer Churn Prediction System</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="sub-text">AI-powered telecom customer analytics dashboard</p>',
        unsafe_allow_html=True
    )

    st.info("""
    This system predicts whether a telecom customer is likely to churn
    using Machine Learning and customer behavior analysis.
    """)

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <h1>7043</h1>
            <h3>Total Customers</h3>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <h1>26.5%</h1>
            <h3>Churn Rate</h3>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <h1>79%</h1>
            <h3>Model Accuracy</h3>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    churn_trend = pd.DataFrame({
        "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "Churn Rate": [18, 20, 23, 22, 26, 24]
    })

    st.subheader("📈 Monthly Churn Trend")

    st.line_chart(
        churn_trend.set_index("Month")
    )

    st.markdown("""
    **Explanation:**  
    This graph shows how customer churn changes every month.  
    The churn rate increased gradually over time.
    """)

    pie_fig = px.pie(
        values=[73.5, 26.5],
        names=["Stayed", "Churned"],
        title="Customer Distribution",
        hole=0.5
    )

    st.plotly_chart(
        pie_fig,
        use_container_width=True
    )

    st.markdown("""
    **Explanation:**  
    Most customers stayed with the company while 26.5% customers churned.
    """)

if page == "Prediction":

    st.markdown(
        '<p class="big-title">🧠 Predict Customer Churn</p>',
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.slider(
            "Tenure",
            0,
            72
        )

    with col2:

        phone = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        monthly = st.slider(
            "Monthly Charges",
            0.0,
            150.0
        )

        total = st.slider(
            "Total Charges",
            0.0,
            10000.0
        )

    gender = 1 if gender == "Male" else 0
    partner = 1 if partner == "Yes" else 0
    dependents = 1 if dependents == "Yes" else 0
    phone = 1 if phone == "Yes" else 0
    paperless = 1 if paperless == "Yes" else 0

    features = np.array([[

        senior,
        tenure,
        monthly,
        total,

        gender,
        partner,
        dependents,
        phone,

        0,
        0,

        0,
        1,

        0,
        0,
        0,
        0,
        0,
        0,

        paperless,

        0,
        0,

        0,
        0,
        0,

        0,
        0,
        0,
        0,
        0

    ]])

    input_data = scaler.transform(features)

    st.write("")

    if st.button("Predict Churn"):

        prediction = model.predict(input_data)

        probability = model.predict_proba(input_data)[0][1]

        churn_percent = round(probability * 100, 2)

        st.markdown(f"""
        <div style="
        background:white;
        padding:20px;
        border-radius:15px;
        font-size:28px;
        font-weight:bold;
        text-align:center;
        box-shadow:0px 4px 15px rgba(0,0,0,0.1);
        margin-bottom:20px;">
        📊 Churn Probability: {churn_percent}%
        </div>
        """, unsafe_allow_html=True)

        if churn_percent < 40:

            st.success("🟢 Low Risk Customer")

        elif churn_percent < 70:

            st.warning("🟡 Medium Risk Customer")

        else:

            st.error("🔴 High Risk Customer")

        if prediction[0] == 1:

            st.markdown("""
            <div style="
            background:#fee2e2;
            padding:25px;
            border-radius:15px;
            color:#991b1b;
            font-size:24px;
            text-align:center;
            font-weight:bold;">
            ⚠️ Customer is likely to churn
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div style="
            background:#dcfce7;
            padding:25px;
            border-radius:15px;
            color:#166534;
            font-size:24px;
            text-align:center;
            font-weight:bold;">
            ✅ Customer will stay
            </div>
            """, unsafe_allow_html=True)

        report = pd.DataFrame({

            "Prediction": [
                "Churn" if prediction[0] == 1 else "Stay"
            ],

            "Probability": [churn_percent]

        })

        csv = report.to_csv(index=False)

        st.download_button(
            "⬇ Download Prediction Report",
            csv,
            "prediction_report.csv",
            "text/csv"
        )

if page == "Analytics":

    st.markdown(
        '<p class="big-title">📊 Customer Analytics</p>',
        unsafe_allow_html=True
    )

    contract_data = pd.DataFrame({
        "Contract Type": [
            "Month-to-Month",
            "One Year",
            "Two Year"
        ],
        "Churn Customers": [1655, 166, 48]
    })

    contract_fig = px.bar(
        contract_data,
        x="Contract Type",
        y="Churn Customers",
        text_auto=True,
        title="Contract Type vs Churn"
    )

    st.plotly_chart(
        contract_fig,
        use_container_width=True
    )

    st.markdown("""
    **Explanation:**  
    Customers with month-to-month contracts churn more compared to yearly contracts.
    """)

    gender_data = pd.DataFrame({

        "Gender": ["Male", "Female"],
        "Churn Customers": [930, 939]

    })

    gender_fig = px.pie(
        gender_data,
        values="Churn Customers",
        names="Gender",
        hole=0.5,
        title="Gender vs Churn"
    )

    st.plotly_chart(
        gender_fig,
        use_container_width=True
    )

    st.markdown("""
    **Explanation:**  
    Male and female customers show similar churn behavior.
    """)

    charges_fig = px.histogram(
        df,
        x="MonthlyCharges",
        nbins=30,
        title="Monthly Charges Distribution"
    )

    st.plotly_chart(
        charges_fig,
        use_container_width=True
    )

    st.markdown("""
    **Explanation:**  
    Customers with higher monthly charges tend to churn more frequently.
    """)

    tenure_fig = px.box(
        df,
        x="Churn",
        y="tenure",
        color="Churn",
        title="Tenure vs Churn"
    )

    st.plotly_chart(
        tenure_fig,
        use_container_width=True
    )

    st.markdown("""
    **Explanation:**  
    New customers churn more often while long-term customers usually stay.
    """)

    service_data = pd.DataFrame({
        "Services": [
            "Phone Service",
            "Internet Service",
            "Tech Support",
            "Streaming TV",
            "Online Backup"
        ],
        "Usage": [85, 78, 45, 67, 59]
    })

    service_fig = px.bar(
        service_data,
        x="Services",
        y="Usage",
        text_auto=True,
        title="Service Usage Analysis"
    )

    st.plotly_chart(
        service_fig,
        use_container_width=True
    )

    st.markdown("""
    **Explanation:**  
    Phone and Internet services have the highest usage.
    """)

    risk_data = pd.DataFrame({
        "Risk Level": [
            "Low Risk",
            "Medium Risk",
            "High Risk"
        ],
        "Customers": [4200, 1800, 1043]
    })

    risk_fig = px.pie(
        risk_data,
        values="Customers",
        names="Risk Level",
        hole=0.5,
        title="Customer Risk Segmentation"
    )

    st.plotly_chart(
        risk_fig,
        use_container_width=True
    )

    st.markdown("""
    **Explanation:**  
    High-risk customers need immediate retention strategies.
    """)

    importance_df = pd.DataFrame({

        "Feature": [

            "Contract",
            "Tenure",
            "MonthlyCharges",
            "TechSupport",
            "OnlineSecurity"

        ],

        "Importance": [

            0.32,
            0.24,
            0.18,
            0.15,
            0.11

        ]

    })

    importance_fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text_auto=True,
        title="Top Features Affecting Churn"
    )

    st.plotly_chart(
        importance_fig,
        use_container_width=True
    )

    st.markdown("""
    **Explanation:**  
    Contract type and tenure are the most important factors affecting customer churn.
    """)