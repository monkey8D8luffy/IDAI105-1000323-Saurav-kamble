# 🛍️ Beyond Discounts: Black Friday Sales Intelligence v4.0

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?style=for-the-badge&logo=plotly)

**Live Application:** https://idai105-1000323-saurav-kamble-a8g7ogb8cpzmabul5jbyn8.streamlit.app/

An enterprise-grade, AI-powered analytics dashboard designed to extract actionable business intelligence from Black Friday retail transaction data. Built with a custom **Liquid Glassmorphism UI**, this application transforms raw retail data into an interactive, multi-dimensional workspace featuring real-time data filtering and embedded machine learning models.

---

## 🚀 Key Features & Analytics

The dashboard is structured into six advanced analytical tiers:

* 📈 **Executive Overview:** High-level liquidity tracking, demographic velocity, and a dynamic **Pareto (80/20) Distribution** analysis highlighting entity concentration.
* 🧠 **Customer Intelligence:** Customer segmentation using an unsupervised **K-Means Clustering** algorithm applied to RFM (Recency, Frequency, Monetary) metrics to map user spending volatility.
* 📦 **Product Analytics:** Hierarchical macro-product topologies using interactive Treemaps and category price dispersion tracking.
* 🚨 **Risk & Anomaly Detection:** Statistical outlier detection using the **Interquartile Range (IQR)** method to isolate extreme high-value transactions from normal frequency bounds.
* 🔬 **Advanced Analytics:** Multivariate Pearson Eigen-Matrices (correlation) and 2D heatmaps tracking the intersection of age, location, and purchase volume.
* 🔮 **Prediction Engine:** A supervised **Random Forest Regression** model trained to forecast a customer's monetary expenditure capacity based on arbitrary demographic input vectors.

---

## 🎨 UI/UX Design

* **Theme:** Liquid Glassmorphism (Crypto-Exchange Navy / Cyan).
* **CSS Architecture:** Fully customized Streamlit UI using raw CSS injection. Features animated noise grain overlays, translucent frosted glass containers (`backdrop-filter: blur`), floating liquid morph animations on hover, and custom WebKit scrollbars.
* **Data Visualization:** Exclusively powered by `Plotly` (Express & Graph Objects) with transparent backgrounds to seamlessly blend into the glass UI.

---

## 🛠️ Technology Stack

* **Frontend & Framework:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`KMeans`, `RandomForestRegressor`, `StandardScaler`)
* **Data Visualization:** Plotly

---

## 💻 Installation & Local Deployment

To run this dashboard locally on your machine, follow these steps:

**1. Clone the repository**
```bash
git clone [https://github.com/YOUR-GITHUB-USERNAME/YOUR-REPOSITORY-NAME.git](https://github.com/YOUR-GITHUB-USERNAME/YOUR-REPOSITORY-NAME.git)
cd YOUR-REPOSITORY-NAME
