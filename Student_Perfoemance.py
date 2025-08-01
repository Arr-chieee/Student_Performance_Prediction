import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor (Made by: Archit Raj)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🎓 Student Performance Predictor</h1>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the student data"""
    try:
        # Try to load the CSV file
        data = pd.read_csv('Simple 100 Student Marks.csv')
        return data
    except FileNotFoundError:
        # If file not found, create sample data based on provided structure
        np.random.seed(42)
        sample_data = {
            'student_name': [f'Student_{i}' for i in range(100)],
            'maths_marks': np.random.randint(0, 101, 100),
            'science_marks': np.random.randint(0, 101, 100),
            'english_marks': np.random.randint(0, 101, 100),
            'social_studies_marks': np.random.randint(0, 101, 100),
            'language_marks': np.random.randint(0, 101, 100)
        }
        return pd.DataFrame(sample_data)

def preprocess_data(df):
    """Preprocess data and create target variable"""
    # Calculate total marks and average
    df['total_marks'] = df[['maths_marks', 'science_marks', 'english_marks', 
                           'social_studies_marks', 'language_marks']].sum(axis=1)
    df['average_marks'] = df['total_marks'] / 5
    
    # Create pass/fail target (passing threshold: 40%)
    df['pass_fail'] = (df['average_marks'] >= 40).astype(int)
    df['result'] = df['pass_fail'].map({1: 'Pass', 0: 'Fail'})
    
    return df

@st.cache_data
def train_model(df):
    """Train the prediction model"""
    # Features for training
    features = ['maths_marks', 'science_marks', 'english_marks', 
                'social_studies_marks', 'language_marks']
    X = df[features]
    y = df['pass_fail']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, accuracy, X_test, y_test, y_pred

def main():
    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["📊 Data Overview", "📈 Analysis & Visualization", 
                                "🤖 Model Training", "🔮 Make Predictions"])
    
    if page == "📊 Data Overview":
        st.header("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Students", len(df))
        with col2:
            st.metric("Subjects", 5)
        with col3:
            pass_rate = (df['pass_fail'].sum() / len(df)) * 100
            st.metric("Pass Rate", f"{pass_rate:.1f}%")
        with col4:
            avg_score = df['average_marks'].mean()
            st.metric("Average Score", f"{avg_score:.1f}")
        
        st.subheader("Sample Data")
        st.dataframe(df.head(10))
        
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())
    
    elif page == "📈 Analysis & Visualization":
        st.header("Data Analysis & Visualization")
        
        
        st.subheader("Subject-wise Performance Distribution")
        subjects = ['maths_marks', 'science_marks', 'english_marks', 
                   'social_studies_marks', 'language_marks']
        
        
        fig_box = go.Figure()
        for subject in subjects:
            fig_box.add_trace(go.Box(y=df[subject], name=subject.replace('_marks', '').title()))
        
        fig_box.update_layout(title="Subject-wise Score Distribution", 
                             yaxis_title="Marks", height=500)
        st.plotly_chart(fig_box, use_container_width=True)
        
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Pass/Fail Distribution")
            pass_fail_counts = df['result'].value_counts()
            fig_pie = px.pie(values=pass_fail_counts.values, 
                            names=pass_fail_counts.index,
                            title="Pass/Fail Distribution",
                            color_discrete_map={'Pass': '#2E8B57', 'Fail': '#DC143C'})
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("Average Marks Distribution")
            fig_hist = px.histogram(df, x='average_marks', nbins=20,
                                   title="Distribution of Average Marks",
                                   color_discrete_sequence=['#1f77b4'])
            fig_hist.add_vline(x=40, line_dash="dash", line_color="red",
                              annotation_text="Pass Threshold (40%)")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        
        st.subheader("Subject Correlation Heatmap")
        corr_matrix = df[subjects].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                            title="Correlation between Subjects")
        st.plotly_chart(fig_corr, use_container_width=True)
        
        
        st.subheader("Subject-wise Performance Analysis")
        subject_stats = []
        for subject in subjects:
            avg_score = df[subject].mean()
            pass_count = (df[subject] >= 40).sum()
            pass_rate = (pass_count / len(df)) * 100
            subject_stats.append({
                'Subject': subject.replace('_marks', '').title(),
                'Average Score': round(avg_score, 2),
                'Pass Rate (%)': round(pass_rate, 2)
            })
        
        stats_df = pd.DataFrame(subject_stats)
        
        fig_bar = px.bar(stats_df, x='Subject', y='Average Score',
                        title="Average Scores by Subject",
                        color='Average Score',
                        color_continuous_scale='viridis')
        st.plotly_chart(fig_bar, use_container_width=True)
    
    elif page == "🤖 Model Training":
        st.header("Model Training & Evaluation")
        
        # Train the model
        with st.spinner("Training model..."):
            model, scaler, accuracy, X_test, y_test, y_pred = train_model(df)
        
        st.success(f"Model trained successfully! Accuracy: {accuracy:.2%}")
        
        # Model performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Performance")
            st.metric("Accuracy", f"{accuracy:.2%}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                              title="Confusion Matrix",
                              labels=dict(x="Predicted", y="Actual"),
                              x=['Fail', 'Pass'], y=['Fail', 'Pass'])
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            st.subheader("Feature Importance")
            feature_names = ['Maths', 'Science', 'English', 'Social Studies', 'Language']
            importance = model.feature_importances_
            
            fig_importance = px.bar(x=feature_names, y=importance,
                                   title="Feature Importance",
                                   labels={'x': 'Subjects', 'y': 'Importance'})
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Classification Report
        st.subheader("Detailed Classification Report")
        report = classification_report(y_test, y_pred, target_names=['Fail', 'Pass'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
    
    elif page == "🔮 Make Predictions":
        st.header("Predict Student Performance")
        
        # Train model for prediction
        model, scaler, accuracy, _, _, _ = train_model(df)
        
        st.info(f"Using trained model with {accuracy:.2%} accuracy")
        
        # Input form
        st.subheader("Enter Student Marks")
        
        col1, col2 = st.columns(2)
        
        with col1:
            maths = st.slider("Mathematics Marks", 0, 100, 50)
            science = st.slider("Science Marks", 0, 100, 50)
            english = st.slider("English Marks", 0, 100, 50)
        
        with col2:
            social = st.slider("Social Studies Marks", 0, 100, 50)
            language = st.slider("Language Marks", 0, 100, 50)
        
        # Prediction button
        if st.button("Predict Performance", type="primary"):
            # Prepare input data
            input_data = np.array([[maths, science, english, social, language]])
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Calculate average
            average = (maths + science + english + social + language) / 5
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Marks", f"{average:.1f}")
            
            with col2:
                result = "Pass" if prediction == 1 else "Fail"
                color = "normal" if prediction == 1 else "inverse"
                st.metric("Prediction", result)
            
            with col3:
                confidence = max(prediction_proba) * 100
                st.metric("Confidence", f"{confidence:.1f}%")
            
            # Visualization
            st.subheader("Subject Performance Breakdown")
            subjects = ['Maths', 'Science', 'English', 'Social Studies', 'Language']
            marks = [maths, science, english, social, language]
            
            fig = px.bar(x=subjects, y=marks, 
                        title="Individual Subject Performance",
                        color=marks,
                        color_continuous_scale='RdYlGn')
            fig.add_hline(y=40, line_dash="dash", line_color="red",
                         annotation_text="Pass Threshold")
            st.plotly_chart(fig, use_container_width=True)
            
            
            prob_df = pd.DataFrame({
                'Outcome': ['Fail', 'Pass'],
                'Probability': prediction_proba
            })
            
            fig_prob = px.bar(prob_df, x='Outcome', y='Probability',
                             title="Prediction Probabilities",
                             color='Outcome',
                             color_discrete_map={'Pass': '#2E8B57', 'Fail': '#DC143C'})
            st.plotly_chart(fig_prob, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("**Student Performance Predictor** - Built with Streamlit & Machine Learning")

if __name__ == "__main__":
    main()