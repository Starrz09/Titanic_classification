import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid;
        margin: 1rem 0;
    }
    .survived-card {
        border-left-color: #28a745;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    }
    .died-card {
        border-left-color: #dc3545;
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    }
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .error-box {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Load LightGBM model with better error handling
@st.cache_resource
def load_model():
    try:
        # Try to import lightgbm first
        import lightgbm as lgb
    except ImportError:
        st.error("LightGBM is not installed. Please add 'lightgbm' to your requirements.txt file.")
        return None
    
    try:
        with open("tuned_lgbm_model.pkl", "rb") as file:
            model = pickle.load(file)
            st.success("✅ Model loaded successfully!")
            return model
    except FileNotFoundError:
        st.error("Model file 'tuned_lgbm_model.pkl' not found. Please ensure the model file is in the repository root directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Check if model can be loaded
model = load_model()

# Header
st.markdown("""
<div class="main-header">
    <h1>🚢 Titanic Survival Predictor</h1>
    <p>Discover your fate aboard the RMS Titanic using a tuned LightGBM model</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", 
             caption="RMS Titanic", use_column_width=True)
    st.markdown("### About the Titanic")
    st.markdown("""
    The RMS Titanic sank in 1912 during its maiden voyage. 
    This app predicts survival chances using historical data and machine learning.
    """)
    st.markdown("### Key Factors")
    st.markdown("""
    - **Passenger Class**
    - **Gender & Age**
    - **Family Size**
    - **Port of Embarkation**
    """)

# If model is not available, show error message but continue with demo functionality
if model is None:
    st.markdown("""
    <div class="error-box">
        <h4>⚠️ Model Not Available</h4>
        <p>The trained model couldn't be loaded. This could be because:</p>
        <ul>
            <li>The model file 'tuned_lgbm_model.pkl' is missing from the repository</li>
            <li>LightGBM dependency is not installed</li>
            <li>There's a compatibility issue with the model file</li>
        </ul>
        <p><strong>You can still explore the app interface and statistics below.</strong></p>
    </div>
    """, unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Make Prediction", "📊 Survival Statistics", "ℹ️ About"])

with tab1:
    st.markdown("### Enter Passenger Details")
    
    if model is None:
        st.warning("⚠️ Model not available - predictions will use demo logic")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Officer", "Noble"])
            sex = st.selectbox("Gender", ["Male", "Female"])
        with col2:
            age = st.slider("Age", 0, 80, 25)
            pclass = st.selectbox("Passenger Class", ["1", "2", "3"])
        with col3:
            fare = st.number_input("Fare Paid (£)", 0.0, 600.0, 32.0)
            embarked = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])

        st.markdown("#### 👨‍👩‍👧‍👦 Family Information")
        col4, col5 = st.columns(2)
        with col4:
            sibsp = st.slider("Siblings/Spouses Aboard", 0, 8, 0)
        with col5:
            parch = st.slider("Parents/Children Aboard", 0, 6, 0)

        submitted = st.form_submit_button("🔮 Predict My Fate")

    if submitted:
        # Feature engineering
        family_size = sibsp + parch
        is_alone = 1 if family_size == 0 else 0
        log_fare = np.log1p(fare)

        # Age group
        if age < 10:
            age_group = 'Child'
        elif age < 20:
            age_group = 'Teenager'
        elif age < 30:
            age_group = 'Young Adult'
        elif age < 40:
            age_group = 'Adult'
        elif age < 60:
            age_group = 'Middle-Aged'
        else:
            age_group = 'Elderly'

        age_class = f"{age_group}_{pclass}"
        sex_pclass = f"{sex}_{pclass}"

        input_data = pd.DataFrame([{
            'siblings_or_spouses_aboard': sibsp,
            'parents_or_children_aboard': parch,
            'log_fare': log_fare,
            'sex': sex,
            'embarked': embarked,
            'p_class': pclass,
            'title': title,
            'age_group': age_group,
            'is_alone': is_alone,
            'age_class': age_class,
            'sex_pclass': sex_pclass
        }])

        if model is not None:
            # Use actual model prediction
            try:
                categorical_cols = ['p_class', 'sex', 'embarked', 'title', 'age_group',
                                    'is_alone', 'age_class', 'sex_pclass']
                input_data[categorical_cols] = input_data[categorical_cols].astype('category')

                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0][1]

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.write("Input data that caused the error:")
                st.dataframe(input_data)
                st.stop()
        else:
            # Use simple rule-based prediction as fallback
            st.info("Using simplified prediction rules (model not available)")
            
            # Simple survival probability based on historical patterns
            base_prob = 0.32  # Overall survival rate
            
            # Gender factor (strongest predictor)
            if sex == "Female":
                base_prob *= 2.3  # Women had much higher survival rates
            else:
                base_prob *= 0.6  # Men had lower survival rates
            
            # Class factor
            if pclass == "1":
                base_prob *= 1.6
            elif pclass == "2":
                base_prob *= 1.2
            else:
                base_prob *= 0.8
            
            # Age factor
            if age < 16:
                base_prob *= 1.3  # Children had higher survival rates
            elif age > 60:
                base_prob *= 0.8  # Elderly had lower survival rates
            
            # Family size factor
            if family_size > 0 and family_size < 4:
                base_prob *= 1.1  # Small families had slight advantage
            elif family_size >= 4:
                base_prob *= 0.9  # Large families had disadvantage
            
            proba = min(base_prob, 0.95)  # Cap at 95%
            prediction = 1 if proba > 0.5 else 0

        st.markdown("---")
        st.markdown("### 🎯 Your Fate Revealed")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Survival Probability", f"{proba*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Family Size", family_size)
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Age Group", age_group)
            st.markdown('</div>', unsafe_allow_html=True)

        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-card survived-card">
                <h2>🎉 Congratulations! You Survived!</h2>
                <p style="font-size: 1.2em;">Your predicted survival probability is <strong>{proba*100:.1f}%</strong>.</p>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
            <div class="prediction-card died-card">
                <h2>💔 Unfortunately, You Did Not Survive</h2>
                <p style="font-size: 1.2em;">Predicted confidence of non-survival: <strong>{(1 - proba)*100:.1f}%</strong>.</p>
            </div>
            """, unsafe_allow_html=True)

        # Survival probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=proba*100,
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgray"},
                    {'range': [25, 50], 'color': "gray"},
                    {'range': [50, 75], 'color': "lightgreen"},
                    {'range': [75, 100], 'color': "green"},
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'value': 90}
            },
            title={'text': "Survival Probability (%)"}
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("🔍 View Feature Values Used"):
            st.dataframe(input_data)

with tab2:
    st.markdown("### 📊 Historical Survival Statistics")
    col1, col2 = st.columns(2)
    with col1:
        class_survival = pd.DataFrame({
            'Class': ['1st Class', '2nd Class', '3rd Class'],
            'Survival Rate': [62.96, 47.28, 24.24]
        })
        fig = px.bar(class_survival, x='Class', y='Survival Rate',
                     color='Survival Rate', color_continuous_scale='RdYlGn',
                     title="Survival Rate by Passenger Class")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        gender_survival = pd.DataFrame({
            'Gender': ['Female', 'Male'],
            'Survival Rate': [74.20, 18.89]
        })
        fig = px.pie(gender_survival, values='Survival Rate', names='Gender',
                     title="Survival Rate by Gender")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Age Group Survival Rates")
    age_survival = pd.DataFrame({
        'Age Group': ['Child', 'Teenager', 'Young Adult', 'Adult', 'Middle-Aged', 'Elderly'],
        'Survival Rate': [61.3, 40.2, 35.8, 36.9, 40.4, 22.7]
    })
    fig = px.bar(age_survival, x='Age Group', y='Survival Rate',
                 color='Survival Rate', color_continuous_scale='RdYlGn',
                 title="Survival Rate by Age Group")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### ℹ️ About This App")
    st.markdown("""
    <div class="info-box">
    <h4>How It Works</h4>
    <p>This application uses a LightGBM model trained on Titanic data. 
    It considers gender, age, family size, ticket fare, and travel class to estimate your chances of survival.</p>
    </div>

    <div class="info-box">
    <h4>Model Performance</h4>
    <ul>
        <li>Train Accuracy: ~92%</li>
        <li>Validation Accuracy: ~85%</li>
        <li>Model: LightGBM Classifier</li>
    </ul>
    </div>

    <div class="info-box">
    <h4>Troubleshooting</h4>
    <p>If the model fails to load, ensure:</p>
    <ul>
        <li>'lightgbm' is in your requirements.txt</li>
        <li>'tuned_lgbm_model.pkl' is in your repository root</li>
        <li>The model was saved with a compatible LightGBM version</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Passengers", "2,224")
    with col2: st.metric("Survivors", "710")
    with col3: st.metric("Survival Rate", "32%")
    with col4: st.metric("♀ Survival Rate", "74%")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Built with Streamlit • Powered by LightGBM</div>", 
    unsafe_allow_html=True
)
