# Flipkart Customer Service Satisfaction — Classification Project

## 📌 Project Summary

In the highly competitive e-commerce space, delivering excellent customer service is crucial for sustaining growth and customer loyalty. Flipkart, as one of the largest e-commerce platforms, focuses on enhancing customer satisfaction to differentiate itself from competitors.

This project analyzes ~22,430 customer service interactions across support channels to identify key drivers of Customer Satisfaction (CSAT), evaluate performance across service teams, and build a machine learning model to predict at-risk (dissatisfied) customers — enabling proactive intervention rather than reactive damage control.

## 🎯 Problem Statement

- **Company:** Flipkart (major e-commerce platform)
- **Objective:** Identify key drivers of CSAT, evaluate agent/team performance, and develop data-driven strategies to improve the support experience
- **Expected Outcomes:** Faster issue resolution, tailored support strategies, optimized agent performance, improved CSAT, and increased customer retention

## 📊 Dataset

The dataset contains 22,430 customer service records with 20 original features, including:

| Category | Columns |
|---|---|
| Interaction Info | `channel_name`, `category`, `Sub-category`, `Customer Remarks` |
| Order Info | `Order_id`, `order_date_time`, `Product_category`, `Item_price` |
| Timing | `Issue_reported at`, `issue_responded`, `Survey_response_Date` |
| Agent Info | `Agent_name`, `Supervisor`, `Manager`, `Tenure Bucket`, `Agent Shift` |
| Target | `CSAT Score` (1–5) |

## 🛠️ Tech Stack

- **Pandas / NumPy** — data manipulation and cleaning
- **Matplotlib / Seaborn** — data visualization
- **Scikit-learn** — model building, tuning, and evaluation
- **TextBlob** — sentiment analysis on customer remarks
- **SHAP** — model explainability
- **SciPy** — statistical hypothesis testing

## 🔍 Project Workflow

### 1. Data Cleaning & Wrangling
- Dropped columns with extreme missingness (`connected_handling_time`, `order_date_time`, `Customer_City`, `Item_price`, `Product_category`) while preserving signal via a `has_order_info` flag
- Dropped high-cardinality/redundant columns (`Agent_name`, `Manager`)
- Removed duplicates (none found)

### 2. Target Engineering
- Reframed the original 1–5 `CSAT Score` (heavily imbalanced: 68.5% scored 5) into a **binary target**: `Satisfied` (CSAT 4–5) vs `Not Satisfied` (CSAT 1–3) → 81.2% / 18.8% split

### 3. Feature Engineering
- **Response time** (`response_time_min`) from timestamp differences — capped at the 99th percentile and log-transformed to handle heavy right-skew
- **Text features** from `Customer Remarks`: sentiment scoring (TextBlob), missing-remark flag, and keyword flags for common complaint terms (refund, delay, worst, poor, rude, wrong, broken/damaged)
- **Categorical encoding:** one-hot encoding for low-cardinality columns, frequency encoding for medium-cardinality columns (`Sub-category`, `Supervisor`) to avoid dimensionality issues and data leakage

### 4. Exploratory Data Analysis
Visualized relationships between satisfaction and: channel, category, response time, agent shift, tenure, supervisor performance, and remark sentiment — see notebook for full chart-by-chart insights.

### 5. Hypothesis Testing
Statistically validated 3 hypotheses using Welch's t-test and Chi-Square test:
- Response time significantly differs between satisfied/dissatisfied customers
- Satisfaction is significantly associated with support channel
- Remark sentiment significantly differs between satisfied/dissatisfied customers

### 6. Model Building
Trained and hyperparameter-tuned three classification models with stratified 5-fold cross-validation and class-weight balancing:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 71.2% | 0.903 | 0.729 | 0.807 | **0.759** |
| Decision Tree | 76.5% | 0.860 | 0.854 | 0.857 | 0.601 |
| Random Forest | **84.4%** | 0.856 | **0.975** | **0.912** | 0.740 |

### 7. Model Selection
**Logistic Regression was selected as the final model.** Although Random Forest scored higher on accuracy and F1, it only identified 23% of actually dissatisfied customers (recall). Since the business goal is proactively catching at-risk customers, Logistic Regression's higher recall (73%) and best ROC-AUC (0.759) — along with its interpretability — made it the better fit.

### 8. Model Explainability
Used Logistic Regression coefficients and **SHAP (LinearExplainer)** to confirm the top satisfaction drivers: response time, remark sentiment, and order-related context.

## 💡 Key Insights

- Customers are polarized — very few "neutral" ratings; dissatisfaction tends to be strong
- **Response time** is a major lever: dissatisfied customers waited ~4x longer (20 min median) than satisfied ones (5 min)
- **Email** is the weakest-performing channel (70.5% satisfaction vs ~81.5% for Inbound/Outcall)
- **Order Related** and **Cancellation** categories have the lowest satisfaction — core transactional friction points
- Newer agents (On Job Training) and certain supervisors show notably lower satisfaction rates, pointing to training/coaching opportunities
- Only 33% of customers leave written feedback, so relying on remarks alone misses many "silently dissatisfied" customers

## ✅ Conclusion

The analysis gives Flipkart concrete, actionable levers to improve CSAT: enforcing faster response-time SLAs (especially for Email and Order/Cancellation issues), targeted coaching for underperforming supervisors and newer agents, and using remark sentiment as a real-time flag for at-risk customers. The final Logistic Regression model supports proactive identification of dissatisfied customers, directly aligning with Flipkart's goal of improving retention and brand loyalty.

## 🚀 How to Run

```bash
git clone <your-repo-link>
cd <repo-folder>
pip install -r requirements.txt
jupyter notebook Flipkart_Customer_Satisfaction.ipynb
```

**Requirements:** pandas, numpy, matplotlib, seaborn, scikit-learn, textblob, shap, scipy
