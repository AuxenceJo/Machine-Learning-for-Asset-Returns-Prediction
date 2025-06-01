# 📊 Financial ML Dashboard – Shiny App

**Author:** Auxence JOVIGNOT  
**Project:** Machine Learning for Asset Returns Prediction  
**Type:** Interactive Financial Modeling App (Shiny)

---

## 🌐 Overview

This Shiny application implements a complete end-to-end machine learning workflow to predict **1-month asset returns (R1M_Usd)** based on financial features. It includes data cleaning, exploration, modeling, and strategy evaluation using Lasso, Random Forest, and XGBoost.

---

## 📦 Features

- Upload and clean RData financial datasets
- Explore data with correlation plots, histograms, and time series
- Train and compare predictive models (Lasso, RF, XGBoost)
- Evaluate long-short strategies and analyze cumulative returns
- Visual logs and performance metrics for transparency

---

## 🧰 Requirements

To run locally, make sure the following R packages are installed:

```r
install.packages(c(
  "shiny", "shinydashboard", "DT", "plotly", "ggplot2", 
  "dplyr", "tidyr", "lubridate", "glmnet", "ranger", 
  "xgboost", "caret", "shinycssloaders", "shinythemes"
))
```

---

## 🚀 Getting Started

```r
library(shiny)
runApp("path_to_your_app_directory")
```

---

## 📂 Tabs Description

### 🔹 Data Tab
- Upload `.RData` file
- Clean data using NA threshold
- Preview cleaned dataset

### 🔹 Exploration Tab
- Visualize target return distribution
- View correlation matrix
- Explore time series trends

### 🔹 Modeling Tab
- Select ML models: Lasso, RF, XGBoost
- Tune train/test split and feature correlation threshold
- Monitor training logs and performance metrics

🕒 **Training can take 5 to 7 minutes** depending on model and data size.

### 🔹 Strategies Tab
- Generate long/short portfolios
- Customize top/bottom percentile thresholds
- View Sharpe ratio, volatility, and cumulative returns

### 🔹 Results Tab
- Compare R² across models
- Analyze quintile-based strategy performance
- Download final report (PDF/text)

---

## 📈 Strategy Formula

The **Sharpe Ratio** of the long-short portfolio is computed as:

\[
\text{Sharpe Ratio} = \frac{\mathbb{E}[R_{LS}]}{\sqrt{(\sigma^2_{\text{long}} + \sigma^2_{\text{short}})/2}}
\]

Where:
- \( R_{LS} \) is the return spread
- \( \sigma_{\text{long}} \), \( \sigma_{\text{short}} \) are volatilities of long/short legs

---

## 📊 Model Performance Metrics

- **R²**: \( 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2} \)
- **RMSE**: \( \sqrt{ \frac{1}{n} \sum (y - \hat{y})^2 } \)

---

## 📜 Author & Contact

- **Name:** Auxence Jovignot  
- **Email:** *(your-email@example.com)*  
- **GitHub:** [github.com/AuxenceJo](https://github.com/AuxenceJo)  
- **LinkedIn:** *(optional)*

---

## 📎 License

MIT License – feel free to reuse and improve.
