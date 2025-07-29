# 🌍 CO₂ Emissions Dashboard for Africa

This project is a dynamic and interactive **Streamlit-powered dashboard** that visualizes and forecasts **CO₂ emissions across African countries**. It provides actionable insights for **policy makers, entrepreneurs, researchers**, and **climate investors**.

Built with 💡 business users in mind, the dashboard highlights:
- Sector-wise emission breakdowns (transportation, energy, manufacturing, etc.)
- Time-series trends and forecasts (using linear regression)
- Country-specific heatmaps and GDP/population insights
- Clean, attractive visualizations for presentations and decision-making

---

## 📊 Key Features

- 🎯 **Interactive Sector Boxes** – Click on any emission sector to drill down into trends.
- 🧠 **Machine Learning Forecasts** – Predict emissions for the next 3 years using socioeconomic data.
- 📈 **Bar Charts, Pie Charts & Line Graphs** – Beautifully styled visualizations powered by Matplotlib and Seaborn.
- 🔥 **Correlation Heatmaps** – Understand relationships between CO₂ emissions and GDP/population.
- 🪴 **Actionable Recommendations** – Green energy, climate tech, and policy innovation opportunities.
- ☁️ **Live Dataset from Hugging Face** – Auto-loads updated CSV data from the cloud.

---

## 🚀 Live Demo

> Coming soon via [Streamlit Cloud](https://streamlit.io/cloud)

---

## 📦 Tech Stack

- [Streamlit](https://streamlit.io/) – For rapid web app development
- [Pandas](https://pandas.pydata.org/) – Data loading and manipulation
- [Seaborn & Matplotlib](https://seaborn.pydata.org/) – Visualizations
- [Scikit-learn](https://scikit-learn.org/) – Forecasting via Linear Regression
- [Hugging Face Datasets](https://huggingface.co/datasets) – Cloud-hosted dataset

---

## 📁 Dataset Source

The data is hosted on Hugging Face:
- [`co2_Emission_Africa.csv`](https://huggingface.co/spaces/NSamson1/Early-Warning-Airquality/raw/main/co2_Emission_Africa.csv)

Fields include:
- Country, Year, Population, GDP (USD and PPP)
- Emissions by sector (Transportation, Manufacturing, LUCF, etc.)
- Total emissions (with and without LUCF)

---

## 📌 How to Run

1. Clone the repo:

```bash
git clone https://github.com/your-username/co2-emissions-dashboard.git
cd co2-emissions-dashboard
