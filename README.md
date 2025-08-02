## Titanic Survival Predictor

This project applies machine learning to predict whether a passenger would survive the Titanic disaster based on key features such as age, sex, passenger class, title, and family information. The pipeline includes domain-aware preprocessing, feature engineering, model training, and deployment.

The preprocessing involved title-based age imputation, one-hot encoding of categorical variables, selective scaling, and creation of interaction features. Three models were evaluated: Logistic Regression, Gradient Boosting, and LightGBM. Logistic Regression and LightGBM showed the best performance, but LightGBM was ultimately selected for deployment.

Try the app here: [https://titanic-survival-predictor.streamlit.app](https://starrz09-titanic-classification-titanic-app-av2bwb.streamlit.app/)

---

### Key Highlights

- Title-based age imputation for better handling of missing data.
- Custom feature engineering including interaction terms like sex × class and age classes.
- Trained and evaluated models: Logistic Regression, Gradient Boosting, and LightGBM.
- Logistic Regression and LightGBM had the best validation performance.
- LightGBM was selected for deployment due to its balance of accuracy and performance.
- Live deployment using Streamlit for interactive predictions.

---

### Tech Stack

- Python (pandas, scikit-learn, lightgbm)
- Streamlit (for deployment)
- SHAP and Plotly (for planned interpretability and visualization)

---

### Author

Lawal Habeeb  
Pharmacist | Aspiring Data Scientist  
Email: habeebobashola09@gmail.com
