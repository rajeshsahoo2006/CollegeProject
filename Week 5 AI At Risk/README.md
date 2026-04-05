# Week 5: AI in Risk Management - NLP, Forecasting & Loan Approval

## BADM 566 - Application of AI in Risk Management

This assignment explores AI applications in risk management including sentiment analysis, NLP-based text mining, time-series forecasting, and logistic regression for loan approval prediction.

> **Note:** Azure ML and MeaningCloud Add-ins (originally required) are deprecated. All solutions are implemented in **Python** using open-source libraries as replacements.

---

## Questions & Implementations

### Q1: Sentiment Analysis on Barclays Tweets
**File:** `Q1_Sentiment_Analysis.py` | **Output:** `Q1_output.txt`

- **Objective:** Calculate positive, negative sentiments, and sentiment scores for Barclays tweets using Relative Proportional Difference (RPD).
- **Approach:** Uses **TextBlob** (Python NLP library) as a replacement for Azure ML sentiment analysis.
- **Formula:** RPD = (Positive - Negative) / (Positive + Negative), ranging from -1 to +1.
- **Data:** 364 tweets from `Barclays_Tweets.xlsx` → sheet `BarclaysTweets`, column E (`tweet_text`)
- **Output:** Results written back to Excel columns E (positive), F (negative), G (sentiment score)
- **Key Findings:**
  - 118 positive tweets (31.6%), 79 negative (21.2%), 167 neutral (45.9%)
  - Average sentiment score: 0.1039 (slightly positive overall)

### Q2: NLP Analysis - Text Classification, Topics, Clustering & Deep Categorization
**File:** `Q2_NLP_Analysis.py` | **Output:** `Q2_output.txt`

- **Objective:** Perform 4 NLP operations on Barclays tweets as a replacement for MeaningCloud Add-in.
- **Data:** `Barclays_Tweets.xlsx` → sheet `BarclaysTweets_topics`

| Analysis | Method | Key Results |
|----------|--------|-------------|
| **1. Text Classification** | TextBlob sentiment + subjectivity | 54.9% Neutral, 29.9% Positive, 15.1% Negative |
| **2. Topics Extraction** | TF-IDF (sklearn) + TextBlob noun phrases | Top topics: barclays, diamond, libor, scandal, bank |
| **3. Text Clustering** | K-Means (k=5) on TF-IDF vectors | 5 clusters: general chatter, LIBOR news, market manipulation, CEO resignation, Bank of England |
| **4. Deep Categorization** | Keyword-based Basel II classification | 10.7% Clients/Products, 9.9% Execution/Delivery, 4.7% Business Disruption, 3.8% Internal Fraud |

### Q3: Forecasting using Single Exponential Smoothing (SES)
**File:** `Q3_Forecasting_SES.py` | **Output:** `Q3_output.txt`

- **Objective:** Forecast 2016 loss using SES and compute statistical measures.
- **Formula:** F(t+1) = F(t) + α × (A(t) - F(t)), with α = 0.5, initial forecast = 75M
- **Step-by-step calculation:**

| Year | Actual Loss (USD MM) | Forecast (USD MM) |
|------|---------------------|-------------------|
| 2011 | 76 | 75.00 |
| 2012 | 75 | 75.50 |
| 2013 | 74 | 75.25 |
| 2014 | 79 | 74.62 |
| 2015 | 78 | 76.81 |
| **2016** | **?** | **77.41** |

- **Statistical Measures:**
  - Expected Loss (Mean): **76.40 million USD**
  - Standard Deviation: **2.07 million USD**
  - Variance: **4.30 million USD²**

### Q4: Loan Approval Prediction using Logistic Regression
**File:** `Q4_Logistic_Regression.py` | **Output:** `Q4_output.txt`

- **Objective:** Predict loan approval using logistic regression on applicant features.
- **Data:** 598 records from `Week5-Loan_Approval.xlsx` with 11 features
- **Data Cleaning:** Missing values filled with mode (categorical) and median (numerical); categorical variables label-encoded.
- **Model Accuracy:** **81.67%**
- **Most Important Feature:** Credit_History (coefficient: 3.00)
- **Prediction for Given Applicant:**

| Feature | Value |
|---------|-------|
| Gender | Male |
| Married | Yes |
| Dependents | 2 |
| Education | Graduate |
| Self_Employed | No |
| ApplicantIncome | 5000 |
| CoapplicantIncome | 1000 |
| LoanAmount | 800 |
| Loan_Amount_Term | 240 |
| Credit_History | 1 |
| Property_Area | Urban |

**Result: APPROVED (Y)** with 65.4% probability

---

## Data Files

| File | Description |
|------|-------------|
| `Barclays_Tweets.xlsx` | Barclays tweet data with sentiment results (Q1) and NLP analysis results (Q2) |
| `Week5-Loan_Approval.xlsx` | Loan applicant data for logistic regression (Q4) |

## Dependencies

```
pip install textblob scikit-learn pandas numpy openpyxl
python -m textblob.download_corpora
```

## How to Run

```bash
python Q1_Sentiment_Analysis.py
python Q2_NLP_Analysis.py
python Q3_Forecasting_SES.py
python Q4_Logistic_Regression.py
```
