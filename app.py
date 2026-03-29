import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Black Friday Insights", layout="wide")
st.title("🛍️ Beyond Discounts: Black Friday Sales Insights")

# Load Data (Cached for speed)
@st.cache_data
def load_data():
    # Reading directly from the ZIP file you uploaded
    df = pd.read_csv('BlackFriday_Cleaned.zip')
    return df


df = load_data()

st.sidebar.header("Filter Options")
selected_gender = st.sidebar.selectbox("Select Gender (0=Male, 1=Female)", options=["All", 0, 1])

# Apply filter if a specific gender is selected
if selected_gender != "All":
    df = df[df['Gender'] == selected_gender]

st.write("### Dataset Preview")
st.dataframe(df.head())

col1, col2 = st.columns(2)

with col1:
    # [span_1](start_span)Visualization: Purchase by Age[span_1](end_span)
    st.write("### Purchase Distribution by Age")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='Age', y='Purchase', data=df, ax=ax, palette="Set2")
    st.pyplot(fig)

with col2:
    # [span_2](start_span)[span_3](start_span)Visualization: Anomaly Detection[span_2](end_span)[span_3](end_span)
    st.write("### Top Occupations of High Spenders")
    
    # Calculate IQR to find anomalies for the dashboard
    Q1 = df['Purchase'].quantile(0.25)
    Q3 = df['Purchase'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    anomalies = df[df['Purchase'] > upper_bound]
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.countplot(x='Occupation', data=anomalies, order=anomalies['Occupation'].value_counts().index[:10], palette="Reds_r", ax=ax2)
    st.pyplot(fig2)

st.write("### Most Popular Product Categories")
# [span_4](start_span)Visualization: Popular product categories[span_4](end_span)
fig3, ax3 = plt.subplots(figsize=(12, 5))
sns.countplot(x='Product_Category_1', data=df, order=df['Product_Category_1'].value_counts().index, palette='viridis', ax=ax3)
st.pyplot(fig3)
