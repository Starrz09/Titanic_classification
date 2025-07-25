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

# Custom CSS for better styling
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
</style>
""", unsafe_allow_html=True)

# Load the pipeline
@st.cache_resource
def load_model():
    try:
        with open("pipeline_updated.pkl", "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'pipeline.pkl' is in the same directory.")
        return None

pipeline = load_model()

# Header
st.markdown("""
<div class="main-header">
    <h1>🚢 Titanic Survival Predictor</h1>
    <p>Discover your fate aboard the RMS Titanic using machine learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", 
             caption="RMS Titanic, image from Britannica", use_column_width=True)
    
    st.markdown("### About the Titanic")
    st.markdown("""
    The RMS Titanic sank on April 15, 1912, during her maiden voyage. 
    This predictor uses machine learning to estimate survival chances 
    based on passenger characteristics.
    """)
    
    st.markdown("### Key Factors")
    st.markdown("""
    - **Passenger Class**: Higher classes had better survival rates
    - **Age & Gender**: Women and children first policy
    - **Family Size**: Small families had advantages
    - **Fare**: Often correlated with class and cabin location
    """)

# Main content area
if pipeline is None:
    st.stop()

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["🎯 Make Prediction", "📊 Survival Statistics", "ℹ️ About"])

with tab1:
    st.markdown("### Enter Passenger Details")
    
    # Input form with better layout
    with st.form("prediction_form", clear_on_submit=False):
        # Personal Information Section
        st.markdown("#### 👤 Personal Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            title = st.selectbox("Title", 
                               ["Mr", "Mrs", "Miss", "Master", "Officer", "Noble"],
                               help="Social title of the passenger")
            sex = st.selectbox("Gender", ["Male", "Female"])
        
        with col2:
            age = st.slider("Age", 0, 80, 25, 
                          help="Age of the passenger in years")
            pclass = st.selectbox("Passenger Class", ["1", "2", "3"],
                                help="1st = Upper, 2nd = Middle, 3rd = Lower class")
        
        with col3:
            fare = st.number_input("Fare Paid (£)", 
                                 min_value=0.0, max_value=600.0, value=32.0,
                                 help="Ticket price in British Pounds")
            embarked = st.selectbox("Port of Embarkation", 
                                  ["Southampton", "Cherbourg", "Queenstown"],
                                  help="Where the passenger boarded the ship")
        
        # Family Information Section
        st.markdown("#### 👨‍👩‍👧‍👦 Family Information")
        col4, col5 = st.columns(2)
        
        with col4:
            sibsp = st.slider("Siblings/Spouses Aboard", 0, 8, 0,
                            help="Number of siblings or spouses traveling together")
        
        with col5:
            parch = st.slider("Parents/Children Aboard", 0, 6, 0,
                            help="Number of parents or children traveling together")
        
        # Submit button
        st.markdown("---")
        submitted = st.form_submit_button("🔮 Predict My Fate", 
                                        use_container_width=True,
                                        type="primary")

    if submitted:
        # Feature engineering matching the notebook exactly
    
        log_fare = np.log1p(fare)
    
        
        # Age group determination
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
        
        # Composite features
        age_class = f"{age_group}_{pclass}"
        sex_pclass = f"{sex}_{pclass}"
        
        # Build DataFrame with only the features used in training
        # Based on the notebook, these are the features after dropping 'survived', 'fare', 'age', 'family_size', 'is_alone'
        input_data = pd.DataFrame([{
            'siblings_or_spouses_aboard': sibsp,
            'parents_or_children_aboard': parch,
            'log_fare': log_fare
            'sex': sex,
            'embarked': embarked,
            'p_class': pclass,
            'title': title,
            'age_group': age_group,
            'age_class': age_class,
            'sex_pclass': sex_pclass
        }])
        
        # Make prediction
        try:
            # Get prediction and probability
            prediction = pipeline.predict(input_data)[0]
            proba = pipeline.predict_proba(input_data)[0][1]
            
            # Enhanced result display
            st.markdown("---")
            st.markdown("### 🎯 Your Fate Revealed")
            
            # Create columns for metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Survival Probability", f"{proba*100:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Family Size", family_size)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with metric_col3:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Age Group", age_group)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Main prediction result
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-card survived-card">
                    <h2>🎉 Congratulations! You Survived!</h2>
                    <p style="font-size: 1.2em;">The model predicts you would have survived the Titanic disaster with a <strong>{proba*100:.1f}%</strong> probability.</p>
                    <p>Factors working in your favor likely include your passenger class, age, gender, and family situation.</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="prediction-card died-card">
                    <h2>💔 Unfortunately, you did not survive</h2>
                    <p style="font-size: 1.2em;">The model predicts you would not have survived, with a <strong>{(1-proba)*100:.1f}%</strong> confidence in this prediction.</p>
                    <p>Historical factors such as passenger class, gender, and age played crucial roles in survival rates.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = proba * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Survival Probability"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgray"},
                        {'range': [25, 50], 'color': "gray"},
                        {'range': [50, 75], 'color': "lightgreen"},
                        {'range': [75, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show feature values used for prediction (for debugging)
            with st.expander("🔍 View Feature Values Used"):
                st.dataframe(input_data)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("Please check that the model file matches the expected features.")
            with st.expander("Debug Information"):
                st.write("Input data shape:", input_data.shape)
                st.write("Input data columns:", list(input_data.columns))
                st.write("Input data:")
                st.dataframe(input_data)

with tab2:
    st.markdown("### 📊 Historical Survival Statistics")
    
    # Create some sample visualizations (you would use real Titanic data here)
    col1, col2 = st.columns(2)
    
    with col1:
        # Sample survival by class chart
        class_survival = pd.DataFrame({
            'Class': ['1st Class', '2nd Class', '3rd Class'],
            'Survival Rate': [62.96, 47.28, 24.24]
        })
        
        fig = px.bar(class_survival, x='Class', y='Survival Rate',
                    title='Survival Rate by Passenger Class',
                    color='Survival Rate',
                    color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sample survival by gender
        gender_survival = pd.DataFrame({
            'Gender': ['Female', 'Male'],
            'Survival Rate': [74.20, 18.89]
        })
        
        fig = px.pie(gender_survival, values='Survival Rate', names='Gender',
                    title='Survival Rate by Gender')
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional statistics
    st.markdown("### Age Group Survival Rates")
    age_survival = pd.DataFrame({
        'Age Group': ['Child', 'Teenager', 'Young Adult', 'Adult', 'Middle-Aged', 'Elderly'],
        'Survival Rate': [61.3, 40.2, 35.8, 36.9, 40.4, 22.7]
    })
    
    fig = px.bar(age_survival, x='Age Group', y='Survival Rate',
                title='Survival Rate by Age Group',
                color='Survival Rate',
                color_continuous_scale='RdYlGn')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### ℹ️ About This Application")
    
    st.markdown("""
    <div class="info-box">
    <h4>How It Works</h4>
    <p>This application uses a Gradient Boosting Classifier trained on historical Titanic passenger data. 
    The model considers multiple factors including passenger class, age, gender, family size, and fare paid 
    to predict survival probability.</p>
    </div>
    
    <div class="info-box">
    <h4>Model Features</h4>
    <p>The model uses the following features after extensive feature engineering and selection:</p>
    <ul>
        <li><strong>Basic Information:</strong> Siblings/spouses aboard, parents/children aboard</li>
        <li><strong>Categorical Features:</strong> Sex, embarked port, passenger class, title, age group</li>
        <li><strong>Engineered Features:</strong> Log fare, log fare per person, age-class combination, sex-class combination</li>
    </ul>
    <p><em>Note: Age, fare, family size, and is_alone features were dropped during model optimization as they had minimal predictive value after encoding.</em></p>
    </div>
    
    <div class="info-box">
    <h4>Model Performance</h4>
    <ul>
        <li>Training Accuracy: 92.70%</li>
        <li>Test Accuracy: 84.92%</li>
        <li>ROC-AUC Score: 0.87</li>
    </ul>
    </div>
    
    <div class="info-box">
    <h4>Historical Context</h4>
    <p>The Titanic disaster remains one of the most studied maritime accidents. The survival patterns 
    reveal important insights about social class, gender norms, and emergency protocols of the early 20th century.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add some key statistics
    st.markdown("### Key Statistics")
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric("Total Passengers", "2,224")
    with stat_col2:
        st.metric("Survivors", "710")
    with stat_col3:
        st.metric("Overall Survival Rate", "32%")
    with stat_col4:
        st.metric("Women & Children First", "74% ♀ survival")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Built with Streamlit • Based on Gradient Boosting Classifier • Historical data analysis</div>", 
    unsafe_allow_html=True
)
