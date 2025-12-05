"""Customer Segmentation Web Application
Professional Streamlit App with Login, Registration, and ML Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import hashlib
import json
import os

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Customer Segmentation AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING - NEW VIBRANT DESIGN
# =============================================================================

st.markdown("""
<style>
    /* Main Background - Sky Blue Gradient */
    .main {
        background: linear-gradient(135deg, #87CEEB 0%, #4A90E2 25%, #5DADE2 50%, #6BB6FF 75%, #87CEEB 100%);
        animation: gradientShift 15s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4A90E2 0%, #5DADE2 100%);
    }

    /* Navigation Bar */
    .nav-bar {
        background: linear-gradient(90deg, #4A90E2 0%, #5DADE2 100%);
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 8px 20px rgba(74, 144, 226, 0.3);
    }

    /* Buttons - Enhanced Design */
    .stButton>button {
        background: linear-gradient(135deg, #4A90E2 0%, #5DADE2 100%);
        color: white;
        border-radius: 30px;
        padding: 12px 30px;
        font-weight: 700;
        font-size: 1em;
        border: none;
        box-shadow: 0 8px 20px rgba(74, 144, 226, 0.4);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton>button:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 12px 30px rgba(74, 144, 226, 0.5);
        background: linear-gradient(135deg, #5DADE2 0%, #4A90E2 100%);
    }

    .stButton>button:active {
        transform: translateY(-2px) scale(1.02);
    }

    /* Cards - Glassmorphism Effect */
    .card {
        background: rgba(255, 255, 255, 0.95);
        padding: 35px;
        border-radius: 25px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        margin: 20px 0;
        backdrop-filter: blur(15px);
        border: 3px solid rgba(255, 255, 255, 0.5);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 50px rgba(0,0,0,0.2);
    }

    /* Metrics - New Gradient Cards */
    .metric-card {
        background: linear-gradient(135deg, #4A90E2 0%, #5DADE2 100%);
        color: white;
        padding: 30px;
        border-radius: 25px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(74, 144, 226, 0.5);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px) scale(1.03);
        box-shadow: 0 15px 40px rgba(74, 144, 226, 0.6);
    }

    /* Headers - White with Shadow */
    h1 {
        color: white !important;
        font-weight: 800 !important;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.2);
    }
    
    h2, h3 {
        color: #2c3e50;
        font-weight: 700;
    }

    /* Input Fields - Enhanced Borders */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        border-radius: 15px;
        border: 3px solid #5DADE2;
        padding: 12px;
        font-size: 1em;
        transition: all 0.3s ease;
    }

    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
        border-color: #4A90E2;
        box-shadow: 0 0 0 4px rgba(74, 144, 226, 0.3);
        transform: scale(1.02);
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 20px;
        border: 3px dashed #5DADE2;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(255, 255, 255, 0.7) 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border: 2px solid rgba(74, 144, 226, 0.3);
    }

    /* Success/Error Messages */
    .success-msg {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #28a745;
        box-shadow: 0 5px 15px rgba(40, 167, 69, 0.2);
    }

    .error-msg {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #dc3545;
        box-shadow: 0 5px 15px rgba(220, 53, 69, 0.2);
    }

    /* Slider */
    .stSlider>div>div>div>div {
        background: #5DADE2;
    }

    /* Multiselect */
    [data-baseweb="select"] {
        border-radius: 15px;
    }

    /* Download Button */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: 700;
        border: none;
        box-shadow: 0 8px 20px rgba(40, 167, 69, 0.4);
    }

    .stDownloadButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(40, 167, 69, 0.5);
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #4A90E2 0%, #5DADE2 100%);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5DADE2 0%, #4A90E2 100%);
    }

    /* Plotly Charts */
    .js-plotly-plot {
        border-radius: 15px;
        overflow: hidden;
    }

    /* Form Submit Button */
    .stFormSubmitButton>button {
        background: linear-gradient(135deg, #4A90E2 0%, #5DADE2 100%) !important;
        color: white !important;
        border-radius: 30px !important;
        padding: 15px 40px !important;
        font-weight: 700 !important;
        font-size: 1.1em !important;
        border: none !important;
        box-shadow: 0 8px 20px rgba(74, 144, 226, 0.4) !important;
        transition: all 0.4s ease !important;
    }

    .stFormSubmitButton>button:hover {
        transform: translateY(-5px) scale(1.05) !important;
        box-shadow: 0 12px 30px rgba(74, 144, 226, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'page' not in st.session_state:
    st.session_state.page = "Home"
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'kmeans_model' not in st.session_state:
    st.session_state.kmeans_model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'pca_model' not in st.session_state:
    st.session_state.pca_model = None
if 'cluster_summary' not in st.session_state:
    st.session_state.cluster_summary = None

# =============================================================================
# USER AUTHENTICATION FUNCTIONS
# =============================================================================

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from JSON file"""
    if os.path.exists('users.json'):
        with open('users.json', 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to JSON file"""
    with open('users.json', 'w') as f:
        json.dump(users, f, indent=4)

def register_user(username, password, email):
    """Register a new user"""
    users = load_users()
    if username in users:
        return False, "Username already exists!"

    users[username] = {
        'password': hash_password(password),
        'email': email,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    save_users(users)
    return True, "Registration successful! Please login."

def authenticate_user(username, password):
    """Authenticate user login"""
    users = load_users()
    if username in users:
        if users[username]['password'] == hash_password(password):
            return True
    return False

def logout():
    """Logout user"""
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.page = "Home"

# =============================================================================
# NAVIGATION BAR
# =============================================================================

def show_navigation():
    """Display navigation bar with login/logout buttons"""
    col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 1])

    with col1:
        st.markdown("<h3 style='color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>🎯 Customer Segmentation AI</h3>", unsafe_allow_html=True)

    with col2:
        if st.button("🏠 Home", key="nav_home"):
            st.session_state.page = "Home"

    with col3:
        if st.session_state.logged_in:
            if st.button("🔬 Analysis", key="nav_logic"):
                st.session_state.page = "Logic"
        else:
            st.button("🔬 Analysis", key="nav_logic_disabled", disabled=True)

    with col4:
        if st.session_state.logged_in:
            if st.button("🎯 Predict", key="nav_predict"):
                st.session_state.page = "Prediction"
        else:
            st.button("🎯 Predict", key="nav_predict_disabled", disabled=True)

    with col5:
        if not st.session_state.logged_in:
            if st.button("📝 Register", key="nav_register"):
                st.session_state.page = "Register"

    with col6:
        if st.session_state.logged_in:
            if st.button(f"🚪 Logout", key="nav_logout"):
                logout()
                st.rerun()
        else:
            if st.button("🔐 Login", key="nav_login"):
                st.session_state.page = "Login"

    st.markdown("---")

# =============================================================================
# HOME PAGE
# =============================================================================

def show_home_page():
    """Display home page"""
    st.markdown("""
    <div style='text-align: center; padding: 60px 0;'>
        <h1 style='font-size: 4em; color: white; text-shadow: 3px 3px 8px rgba(0,0,0,0.3); animation: fadeIn 1s ease;'>
            🎯 Customer Segmentation AI Platform
        </h1>
        <p style='font-size: 1.8em; color: white; margin-top: 20px; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>
            🚀 Unlock Powerful Customer Insights with Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature Cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='card'>
            <h3 style='text-align: center; color: #4A90E2;'>🔍 Advanced Analytics</h3>
            <p style='text-align: center; color: #666; font-size: 1.1em;'>
                Utilize K-Means clustering and PCA for deep customer insights
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='card'>
            <h3 style='text-align: center; color: #4A90E2;'>📊 Visual Insights</h3>
            <p style='text-align: center; color: #666; font-size: 1.1em;'>
                Interactive visualizations to understand customer segments
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='card'>
            <h3 style='text-align: center; color: #4A90E2;'>🎯 Smart Predictions</h3>
            <p style='text-align: center; color: #666; font-size: 1.1em;'>
                Predict customer segments for targeted marketing strategies
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Key Features
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #4A90E2;'>✨ Key Features</h3>
            <ul style='font-size: 1.2em; color: #555; line-height: 2;'>
                <li>🔐 Secure user authentication</li>
                <li>📈 Real-time customer segmentation</li>
                <li>🎨 Interactive data visualizations</li>
                <li>🤖 ML-powered predictions</li>
                <li>📊 Comprehensive analytics dashboard</li>
                <li>💾 Export reports and insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #4A90E2;'>🚀 Getting Started</h3>
            <ol style='font-size: 1.2em; color: #555; line-height: 2;'>
                <li><b>Register</b> your account</li>
                <li><b>Login</b> to access the platform</li>
                <li><b>Upload</b> your customer data</li>
                <li><b>Analyze</b> customer segments</li>
                <li><b>Predict</b> segments for new customers</li>
                <li><b>Export</b> insights for action</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    # Call to Action
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if not st.session_state.logged_in:
            st.markdown("""
            <div class='card' style='text-align: center; background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(230,245,255,0.95) 100%);'>
                <h2 style='color: #4A90E2;'>Ready to Get Started?</h2>
                <p style='color: #666; font-size: 1.2em; margin: 20px 0;'>Create your account now and start analyzing customer segments!</p>
                <p style='font-size: 2.5em;'>👆</p>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# REGISTRATION PAGE
# =============================================================================

def show_registration_page():
    """Display registration page"""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div class='card'>
            <h2 style='text-align: center; color: #4A90E2;'>📝 Create New Account</h2>
            <p style='text-align: center; color: #666; margin-bottom: 30px;'>Join us today and unlock powerful insights!</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("registration_form"):
            username = st.text_input("👤 Username", placeholder="Enter username")
            email = st.text_input("📧 Email", placeholder="Enter email address")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter password")
            confirm_password = st.text_input("🔒 Confirm Password", type="password", placeholder="Re-enter password")

            col_1, col_2, col_3 = st.columns([1, 2, 1])
            with col_2:
                submit = st.form_submit_button("🚀 Register Account", use_container_width=True)

            if submit:
                if not username or not email or not password:
                    st.error("❌ All fields are required!")
                elif len(password) < 6:
                    st.error("❌ Password must be at least 6 characters!")
                elif password != confirm_password:
                    st.error("❌ Passwords do not match!")
                else:
                    success, message = register_user(username, password, email)
                    if success:
                        st.success(f"✅ {message}")
                        st.balloons()
                        st.info("👉 Click 'Login' button above to sign in")
                    else:
                        st.error(f"❌ {message}")

# =============================================================================
# LOGIN PAGE
# =============================================================================

def show_login_page():
    """Display login page"""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div class='card'>
            <h2 style='text-align: center; color: #4A90E2;'>🔐 Login to Your Account</h2>
            <p style='text-align: center; color: #666; margin-bottom: 30px;'>Welcome back! Please enter your credentials.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")

            col_1, col_2, col_3 = st.columns([1, 2, 1])
            with col_2:
                submit = st.form_submit_button("🚀 Login", use_container_width=True)

            if submit:
                if authenticate_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.page = "Logic"
                    st.success(f"✅ Welcome back, {username}!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password!")

# =============================================================================
# LOGIC/ANALYSIS PAGE
# =============================================================================

def show_logic_page():
    """Display ML analysis page"""
    st.markdown("<h1 style='text-align: center; color: white;'>🔬 Customer Segmentation Analysis</h1>", unsafe_allow_html=True)

    # File Upload
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📁 Upload Customer Data")

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is None:
        st.info("👆 Upload a CSV file to begin analysis, or use sample data below")
        if st.button("🎲 Generate Sample Data"):
            df = generate_sample_data()
            st.session_state.df = df
            st.success("✅ Sample data generated successfully!")
            st.rerun()
    else:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("✅ File uploaded successfully!")

    st.markdown("</div>", unsafe_allow_html=True)

    # If data exists, show analysis
    if 'df' in st.session_state:
        df = st.session_state.df

        # Data Preview
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Data Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", len(df), delta="Active")
        with col2:
            st.metric("Features", len(df.columns), delta="Available")
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum(), delta="Clean" if df.isnull().sum().sum() == 0 else "Check")
        with col4:
            st.metric("Data Points", df.size, delta="Ready")

        st.dataframe(df.head(10), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Model Training Section
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### ⚙️ Model Configuration")

        col1, col2 = st.columns(2)
        with col1:
            features = st.multiselect(
                "Select Features for Clustering",
                options=[col for col in df.columns if df[col].dtype in ['int64', 'float64']],
                default=[col for col in df.columns if df[col].dtype in ['int64', 'float64']][:4]
            )

        with col2:
            n_clusters = st.slider("Number of Clusters", 2, 10, 4)

        if st.button("🚀 Train Model", use_container_width=True):
            if len(features) < 2:
                st.error("❌ Please select at least 2 features!")
            else:
                with st.spinner("Training model... Please wait..."):
                    train_model(df, features, n_clusters)

        st.markdown("</div>", unsafe_allow_html=True)

        # Show results if model is trained
        if st.session_state.model_trained:
            show_analysis_results(df)

def generate_sample_data():
    """Generate sample customer data"""
    np.random.seed(42)
    n = 500

    data = {
        'CustomerID': range(1, n + 1),
        'Age': np.random.randint(18, 70, n),
        'Annual_Income': np.random.randint(15000, 150000, n),
        'Spending_Score': np.random.randint(1, 100, n),
        'Purchase_Frequency': np.random.randint(1, 50, n),
        'Average_Transaction_Value': np.random.randint(20, 500, n),
        'Years_as_Customer': np.random.randint(0, 10, n)
    }

    return pd.DataFrame(data)

def train_model(df, features, n_clusters):
    """Train K-Means clustering model"""
    try:
        X = df[features]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # Apply PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        # Store in session state
        df['Cluster'] = clusters
        df['PCA1'] = X_pca[:, 0]
        df['PCA2'] = X_pca[:, 1]

        st.session_state.df = df
        st.session_state.kmeans_model = kmeans
        st.session_state.scaler = scaler
        st.session_state.pca_model = pca
        st.session_state.features = features
        st.session_state.model_trained = True
        st.session_state.cluster_summary = df.groupby('Cluster')[features].mean()

        st.success("✅ Model trained successfully!")
        st.rerun()

    except Exception as e:
        st.error(f"❌ Error training model: {str(e)}")

def show_analysis_results(df):
    """Display analysis results"""

    # Cluster Distribution
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📈 Cluster Distribution")

    cluster_counts = df['Cluster'].value_counts().sort_index()

    fig = px.bar(
        x=cluster_counts.index,
        y=cluster_counts.values,
        labels={'x': 'Cluster', 'y': 'Number of Customers'},
        title='Customer Distribution Across Clusters',
        color=cluster_counts.values,
        color_continuous_scale=['#4A90E2', '#5DADE2', '#6BB6FF', '#87CEEB']
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # PCA Visualization
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🎨 PCA Visualization")

    fig = px.scatter(
        df, x='PCA1', y='PCA2', color='Cluster',
        title='Customer Segments - PCA Visualization',
        labels={'PCA1': 'First Principal Component', 'PCA2': 'Second Principal Component'},
        color_continuous_scale=['#4A90E2', '#5DADE2', '#6BB6FF', '#87CEEB'],
        hover_data=st.session_state.features
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Cluster Characteristics
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Cluster Characteristics")

    st.dataframe(st.session_state.cluster_summary.round(2), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Download Results
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name="customer_segments.csv",
        mime="text/csv"
    )

# =============================================================================
# PREDICTION PAGE
# =============================================================================

def show_prediction_page():
    """Display prediction page"""
    st.markdown("<h1 style='text-align: center; color: white;'>🎯 Customer Segment Prediction</h1>", unsafe_allow_html=True)

    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first in the Analysis page!")
        return

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📝 Enter Customer Information")

    col1, col2 = st.columns(2)

    input_data = {}
    features = st.session_state.features

    for i, feature in enumerate(features):
        with col1 if i % 2 == 0 else col2:
            input_data[feature] = st.number_input(
                f"{feature}",
                value=0.0,
                step=1.0
            )

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔮 Predict Segment", use_container_width=True):
        try:
            # Prepare input
            input_df = pd.DataFrame([input_data])

            # Scale and predict
            input_scaled = st.session_state.scaler.transform(input_df)
            prediction = st.session_state.kmeans_model.predict(input_scaled)[0]

            # PCA transform
            input_pca = st.session_state.pca_model.transform(input_scaled)

            # Display result
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🎊 Prediction Result")

            col1, col2, col3 = st.columns(3)
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <h2>Predicted Cluster</h2>
                    <h1 style='font-size: 5em; margin: 20px 0;'>{prediction}</h1>
                    <p style='font-size: 1.2em;'>✨ Segment Identified!</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### 📋 Cluster Characteristics:")
            cluster_info = st.session_state.cluster_summary.loc[prediction]

            cols = st.columns(len(features))
            for i, (feature, value) in enumerate(cluster_info.items()):
                with cols[i]:
                    st.metric(feature, f"{value:.2f}", delta="Avg")

            st.markdown("</div>", unsafe_allow_html=True)

            st.balloons()

        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application logic"""

    # Show navigation
    show_navigation()

    # Route to appropriate page
    if st.session_state.page == "Home":
        show_home_page()
    elif st.session_state.page == "Register":
        show_registration_page()
    elif st.session_state.page == "Login":
        show_login_page()
    elif st.session_state.page == "Logic":
        if st.session_state.logged_in:
            show_logic_page()
        else:
            st.warning("⚠️ Please login to access this page!")
            show_login_page()
    elif st.session_state.page == "Prediction":
        if st.session_state.logged_in:
            show_prediction_page()
        else:
            st.warning("⚠️ Please login to access this page!")
            show_login_page()

if __name__ == "__main__":
    main()