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

## 🧠 Methodology

The application predicts 1-month returns (`R1M_Usd`) using machine learning models trained on engineered financial factors.

### 📈 Sharpe Ratio Formula

```math
\text{Sharpe Ratio} = \frac{\mathbb{E}[R_{LS}]}{\sqrt{(\sigma_{\text{long}}^2 + \sigma_{\text{short}}^2)/2}}
```

Where:
- `R_{LS}`: long-short return
- `\sigma`: standard deviation of top and bottom groups
---

## 🧑‍💻 Author

**Auxence [@AuxenceJo]**

If you use or modify this project, a ⭐ on GitHub would be appreciated!

If you have any advice please do not hesitate ton contact me via github or [linkedin.com](https://www.linkedin.com/in/auxence-jovignot/)

---

## 📜 License

MIT License. Free to use and adapt with attribution.
