import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
import shap

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
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
        color: #0d47a1;
        box-shadow: 0 2px 8px rgba(33, 150, 243, 0.2);
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
    .explanation-box {
        background: #e7f3ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .feature-impact {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem;
        margin: 0.3rem 0;
        border-radius: 5px;
    }
    .positive-impact {
        background: linear-gradient(90deg, rgba(40, 167, 69, 0.1), rgba(40, 167, 69, 0.05));
        border-left: 3px solid #28a745;
    }
    .negative-impact {
        background: linear-gradient(90deg, rgba(220, 53, 69, 0.1), rgba(220, 53, 69, 0.05));
        border-left: 3px solid #dc3545;
    }
    .algorithm-info {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #9c27b0;
        margin: 1rem 0;
        color: #4a148c;
        box-shadow: 0 2px 8px rgba(156, 39, 176, 0.2);
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

# Initialize SHAP explainer
@st.cache_resource
def get_shap_explainer(_model):
    if _model is not None:
        try:
            explainer = shap.TreeExplainer(_model)
            return explainer
        except Exception as e:
            st.warning(f"Could not initialize SHAP explainer: {str(e)}")
            return None
    return None

# Check if model can be loaded
model = load_model()
explainer = get_shap_explainer(model) if model is not None else None

# Header
st.markdown("""
<div class="main-header">
    <h1>🚢 Titanic Survival Predictor</h1>
    <p>Discover your fate aboard the RMS Titanic with AI-powered predictions and explanations</p>
</div>
""", unsafe_allow_html=True)

# Algorithm Information
st.markdown("""
<div class="algorithm-info">
    <h4>🤖 Machine Learning Algorithm: LightGBM (Light Gradient Boosting Machine)</h4>
    <p><strong>LightGBM</strong> is a high-performance gradient boosting framework that uses tree-based learning algorithms. 
    It's designed to be distributed and efficient with faster training speed, higher efficiency, lower memory usage, 
    and better accuracy than traditional gradient boosting methods.</p>
    <ul>
        <li><strong>Type:</strong> Ensemble Learning (Gradient Boosting)</li>
        <li><strong>Strengths:</strong> Handles categorical features well, fast training, high accuracy</li>
        <li><strong>Perfect for:</strong> Tabular data like passenger information</li>
    </ul>
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
    - **Passenger Class** - Higher class = better survival
    - **Gender** - Women had much higher survival rates
    - **Age** - Children were prioritized in lifeboats
    - **Family Size** - Small families had advantages
    - **Fare** - Higher fare often meant better location on ship
    """)
    
    if model is not None:
        st.success("🤖 LightGBM Model: Active")
        if explainer is not None:
            st.success("🔍 SHAP Explanations: Available")
        else:
            st.warning("🔍 SHAP Explanations: Unavailable")
    else:
        st.error("🤖 LightGBM Model: Unavailable")

# If model is not available, stop the app
if model is None:
    st.markdown("""
    <div class="error-box">
        <h4>⚠️ Model Required</h4>
        <p>The trained LightGBM model couldn't be loaded. This app requires the model to function. Please ensure:</p>
        <ul>
            <li>The model file 'tuned_lgbm_model.pkl' is present in the repository</li>
            <li>LightGBM dependency is installed</li>
            <li>The model file is compatible with the current environment</li>
        </ul>
        <p><strong>The app cannot proceed without the trained model.</strong></p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Main prediction interface
st.markdown("### 🎯 Enter Passenger Details")

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

    submitted = st.form_submit_button("🔮 Predict My Fate", use_container_width=True)

if submitted:
    # Feature engineering to match training data exactly
    family_size = sibsp + parch
    is_alone = 1 if family_size == 0 else 0
    log_fare = np.log1p(fare)
    
    # Calculate log_fare_per_person
    family_size_for_fare = family_size + 1  # +1 for the passenger themselves
    log_fare_per_person = log_fare - np.log1p(family_size_for_fare - 1) if family_size_for_fare > 1 else log_fare

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

    # Create input data with exact same features as training
    input_data = pd.DataFrame([{
        'p_class': pclass,
        'sex': sex,
        'age': age,
        'siblings_or_spouses_aboard': sibsp,
        'parents_or_children_aboard': parch,
        'embarked': embarked,
        'title': title,
        'age_group': age_group,
        'family_size': family_size,
        'is_alone': is_alone,
        'log_fare': log_fare,
        'log_fare_per_person': log_fare_per_person,
        'age_class': age_class,
        'sex_pclass': sex_pclass
    }])

    # Use LightGBM model prediction
    try:
        # Convert categorical columns to 'category' dtype exactly like training
        categorical_cols = ['p_class', 'sex', 'embarked', 'title', 'age_group',
                            'is_alone', 'age_class', 'sex_pclass']
        input_data[categorical_cols] = input_data[categorical_cols].astype('category')

        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.write("Detailed error information:")
        st.write(f"Input data shape: {input_data.shape}")
        st.write("Input data columns:", list(input_data.columns))
        st.write("Input data:")
        st.dataframe(input_data)
        st.stop()

    st.markdown("---")
    st.markdown("### 🎯 Your Fate Revealed")
    
    # Main prediction result
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
            <p><em>Prediction made by LightGBM algorithm</em></p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown(f"""
        <div class="prediction-card died-card">
            <h2>💔 Unfortunately, You Did Not Survive</h2>
            <p style="font-size: 1.2em;">Predicted confidence of non-survival: <strong>{(1 - proba)*100:.1f}%</strong>.</p>
            <p><em>Prediction made by LightGBM algorithm</em></p>
        </div>
        """, unsafe_allow_html=True)

    # Create two columns for visualizations
    col1, col2 = st.columns(2)
    
    with col1:
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
        fig.update_layout(height=400, title="LightGBM Prediction")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Feature importance chart (simplified version based on common patterns)
        feature_importance = {
            'Gender': 0.35 if sex == "Female" else -0.45,
            'Passenger Class': 0.25 if pclass == "1" else (0.1 if pclass == "2" else -0.2),
            'Age': 0.15 if age < 16 else (-0.1 if age > 60 else 0),
            'Family Size': 0.1 if 0 < family_size < 4 else (-0.1 if family_size >= 4 else -0.05),
            'Fare': 0.1 if fare > 50 else (-0.05 if fare < 10 else 0)
        }
        
        features = list(feature_importance.keys())
        impacts = list(feature_importance.values())
        colors = ['green' if x > 0 else 'red' for x in impacts]
        
        fig = go.Figure(go.Bar(
            x=impacts,
            y=features,
            orientation='h',
            marker_color=colors,
            text=[f"{abs(x):.2f}" for x in impacts],
            textposition="auto"
        ))
        fig.update_layout(
            title="General Feature Impact Patterns",
            xaxis_title="Impact (+ helps survival, - hurts survival)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # SHAP Explanations (if available)
    if explainer is not None:
        st.markdown("### 🔍 LightGBM Model Explanation")
        st.markdown("""
        <div class="explanation-box">
            <h4>🧠 Why did the LightGBM algorithm make this prediction?</h4>
            <p>The SHAP (SHapley Additive exPlanations) analysis below shows exactly how each of your characteristics 
            influenced the LightGBM model's decision. Green bars push toward survival, red bars push toward death.</p>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # Get SHAP values
            shap_values = explainer.shap_values(input_data)
            
            # For binary classification, we want the positive class (survival)
            if isinstance(shap_values, list):
                shap_values_survival = shap_values[1]  # Positive class
            else:
                shap_values_survival = shap_values
            
            # Create SHAP waterfall plot
            st.markdown("#### Your Personal Feature Impact Analysis")
            
            # Get feature names and values
            feature_names = input_data.columns.tolist()
            feature_values = input_data.iloc[0].values
            shap_vals = shap_values_survival[0]
            
            # Create a more readable feature impact summary
            feature_impact_data = []
            for i, (name, value, shap_val) in enumerate(zip(feature_names, feature_values, shap_vals)):
                # Make feature names more readable
                readable_name = name.replace('_', ' ').title()
                if readable_name == 'P Class':
                    readable_name = f"Passenger Class ({value})"
                elif readable_name == 'Sex':
                    readable_name = f"Gender ({value})"
                elif readable_name == 'Log Fare':
                    readable_name = f"Fare Level (£{fare:.1f})"
                elif readable_name == 'Is Alone':
                    readable_name = f"Traveling Alone ({'Yes' if value == 1 else 'No'})"
                elif 'Class' in readable_name or 'Pclass' in readable_name:
                    readable_name = f"{readable_name} ({value})"
                else:
                    readable_name = f"{readable_name} ({value})"
                
                feature_impact_data.append({
                    'feature': readable_name,
                    'impact': shap_val,
                    'abs_impact': abs(shap_val)
                })
            
            # Sort by absolute impact
            feature_impact_data.sort(key=lambda x: x['abs_impact'], reverse=True)
            
            # Display top features
            st.markdown("**Most Important Factors for Your Prediction:**")
            for i, item in enumerate(feature_impact_data[:8]):  # Show top 8 features
                impact_val = item['impact']
                if impact_val > 0:
                    impact_text = f"+{impact_val:.3f}"
                    impact_class = "positive-impact"
                    icon = "✅"
                else:
                    impact_text = f"{impact_val:.3f}"
                    impact_class = "negative-impact"
                    icon = "❌"
                
                st.markdown(f"""
                <div class="feature-impact {impact_class}">
                    <span><strong>{icon} {item['feature']}</strong></span>
                    <span><strong>{impact_text}</strong></span>
                </div>
                """, unsafe_allow_html=True)
            
            # Create SHAP bar plot using plotly
            fig = go.Figure()
            
            # Sort features by SHAP value for better visualization
            sorted_indices = np.argsort(shap_vals)
            sorted_features = [feature_names[i] for i in sorted_indices]
            sorted_shap_vals = shap_vals[sorted_indices]
            
            colors = ['red' if x < 0 else 'green' for x in sorted_shap_vals]
            
            fig.add_trace(go.Bar(
                x=sorted_shap_vals,
                y=sorted_features,
                orientation='h',
                marker_color=colors,
                text=[f"{x:.3f}" for x in sorted_shap_vals],
                textposition="auto"
            ))
            
            fig.update_layout(
                title="SHAP Feature Importance (Detailed)",
                xaxis_title="SHAP Value (impact on prediction)",
                yaxis_title="Features",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Base value explanation with improved visibility
            base_value = explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1]  # For binary classification
            
            st.markdown(f"""
            <div class="info-box">
                <h4>📊 Understanding the Numbers</h4>
                <p><strong>How LightGBM made your prediction:</strong></p>
                <ul style="font-size: 1.1em; line-height: 1.6;">
                    <li><strong>Base prediction:</strong> {base_value:.3f} (average survival probability for all passengers)</li>
                    <li><strong>Your final prediction:</strong> {proba:.3f} ({proba*100:.1f}% survival chance)</li>
                    <li><strong>Total SHAP impact:</strong> {np.sum(shap_vals):.3f} (how features changed the base prediction)</li>
                    <li><strong>🟢 Green features</strong> increase your survival chances</li>
                    <li><strong>🔴 Red features</strong> decrease your survival chances</li>
                    <li><strong>Algorithm:</strong> LightGBM (Light Gradient Boosting Machine)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error generating SHAP explanation: {str(e)}")
            st.info("SHAP explanations are not available for this prediction.")
    
    else:
        st.info("🔍 SHAP explanations require the explainer to be properly initialized")

    # Input data summary
    with st.expander("🔍 View Your Input Data"):
        st.dataframe(input_data, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Built with Streamlit • Powered by LightGBM • Explained by SHAP</div>", 
    unsafe_allow_html=True
)
