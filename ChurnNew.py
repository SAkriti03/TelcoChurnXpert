# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 01:38:22 2025

@author: Akriti
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import joblib
import math
#from st_aggrid import GridOptionsBuilder, AgGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

##Set page config
st.set_page_config(page_title="Telco ChurnXpert", layout="wide")

# **🔹 Dark and Light Mode CSS - Enhanced UI**
dark_css = """
    <style>
        body, .stApp { background-color: #121212; color: #e0e0e0; transition: all 0.3s ease-in-out; }
        .sidebar .sidebar-content { background-color: #1f1f1f; }
        h1, h2, h3, h4, h5, h6 { color: #ffcc00 !important; text-transform: uppercase; letter-spacing: 1px; }
        .stButton>button { background-color: #ffcc00; color: #121212; border-radius: 10px; }
        .stTextInput>div>div>input { background-color: #2d2d2d; color: #ffffff; border-radius: 5px; }
    </style>
"""

light_css = """
    <style>
        body, .stApp { background-color: #ffffff; color: #000000; }
        .sidebar .sidebar-content { background-color: #f0f0f0; }
    </style>
"""

# **🔹 Toggle for Dark Mode**
dark_mode = st.toggle("🌙 Enable Dark Mode")
if dark_mode:
    st.markdown(dark_css, unsafe_allow_html=True)
else:
    st.markdown(light_css, unsafe_allow_html=True)


# Custom Styling
st.markdown("""
    <style>
        /* Auto-detect Light/Dark Mode for Sidebar */
        [data-testid="stSidebar"] {
            background-color: var(--background-color) !important;
        }

        /* Sidebar Title */
        [data-testid="stSidebar"] h2 {
            color: var(--primary-color) !important;
            font-size: 22px;
        }

        /* Radio Buttons */
        div[data-testid="stSidebar"] div[role="radiogroup"] label {
            background-color: transparent !important;
            color: var(--text-color) !important;
            border: 2px solid var(--border-color) !important;
            border-radius: 8px;
            padding: 8px 10px;
            margin-bottom: 5px;
            transition: background 0.3s ease-in-out;
        }

        /* Hover Effect */
        div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background-color: var(--hover-background) !important;
            color: var(--hover-text) !important;
        }

        /* Selected Radio Button */
        div[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"] {
            background-color: var(--selected-background) !important;
            color: var(--selected-text) !important;
            border-color: var(--selected-border) !important;
        }

        /* Buttons */
        button {
            background-color: var(--button-background) !important;
            color: var(--button-text) !important;
            border: 2px solid var(--button-border) !important;
            border-radius: 8px;
            padding: 10px 15px;
            transition: background 0.3s ease-in-out;
        }

        /* Button Hover */
        button:hover {
            background-color: var(--button-hover-background) !important;
            color: var(--button-hover-text) !important;
            border-color: var(--button-hover-border) !important;
        }

        /* Define Variables for Light and Dark Mode */
        @media (prefers-color-scheme: dark) {
            :root {
                --background-color: #1e1e1e;
                --primary-color: #ffcc00;
                --text-color: #ffffff;
                --border-color: #ffcc00;
                --hover-background: #444;
                --hover-text: #ffcc00;
                --selected-background: #ffcc00;
                --selected-text: black;
                --selected-border: black;
                --button-background: #444;
                --button-text: #ffffff;
                --button-border: #ffcc00;
                --button-hover-background: #ffcc00;
                --button-hover-text: black;
                --button-hover-border: black;
            }
        }

        @media (prefers-color-scheme: light) {
            :root {
                --background-color: #f4f4f4;
                --primary-color: #333;
                --text-color: #000000;
                --border-color: #333;
                --hover-background: #ddd;
                --hover-text: #333;
                --selected-background: #ffcc00;
                --selected-text: black;
                --selected-border: black;
                --button-background: #ddd;
                --button-text: black;
                --button-border: #333;
                --button-hover-background: #ffcc00;
                --button-hover-text: black;
                --button-hover-border: black;
            }
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        /* Modern Button Styling */
        .stDownloadButton, .stButton>button {
            border: none !important;
            border-radius: 12px !important;
            padding: 12px 24px !important;
            margin: 5px;
            background: linear-gradient(145deg, #4b9ac9, #2a3f5f) !important;
            color: white !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
            position: relative;
            overflow: hidden;
        }

        .stDownloadButton:before, .stButton>button:before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                120deg,
                transparent,
                rgba(255,255,255,0.3),
                transparent
            );
            transition: all 0.5s;
        }

        .stDownloadButton:hover:before, .stButton>button:hover:before {
            left: 100%;
        }

        .stDownloadButton:hover, .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.2) !important;
        }

        .stDownloadButton:active, .stButton>button:active {
            transform: translateY(0);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        }

        /* Secondary Button Style */
        .secondary-btn {
            background: linear-gradient(145deg, #ffffff, #f8f9fa) !important;
            color: #2a3f5f !important;
            border: 1px solid #e0e0e0 !important;
        }

        /* Icon Animation */
        .btn-icon {
            margin-right: 8px;
            transition: transform 0.3s ease;
        }

        .stDownloadButton:hover .btn-icon {
            transform: translateX(3px);
        }
    </style>
""", unsafe_allow_html=True)

# Apply custom styling to enhance visibility on black background
st.markdown("""
<style>
    label {
        color: #ff00ff !important;  /* Neon Pink */
        font-weight: bold;
        font-size: 46px;
    }
</style>
""", unsafe_allow_html=True)

# App Header with a decorative banner image using a local image
col1, col2 = st.columns([1, 5])
with col1:
    st.image("churn1.png", width=200)
with col2:
    st.markdown("""
        <div style="text-align: left; background: linear-gradient(90deg, #3C1053, #3498DB); padding: 20px; border-radius: 10px; margin-bottom: 15px;">
            <h1 style="color: #FFD700; font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 32px; font-weight: 600;">
                Telco ChurnXpert – A smart churn prediction & analytics tool
            </h1>
        </div>
        """, unsafe_allow_html=True)

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    return df

df = load_data()

# Preprocessing function
def preprocess_data(data):
    data = data.copy()
    # Drop customerID column if it exists
    if 'customerID' in data.columns:
        data = data.drop(columns=['customerID'])
   
    # Replace inconsistent service strings
    data.replace({'No internet service': 'No', 'No phone service': 'No'}, inplace=True)
   
    # Convert TotalCharges to numeric and fill missing values with the median
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
   
    # Encode binary categorical variables
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        data[col] = data[col].apply(lambda x: 1 if x in ['Male', 'Yes'] else 0)
   
    # Function to encode three-option services
    def encode_service(x):
        if x == "Yes":
            return 2
        elif x == "No":
            return 0
        else:
            return 1  # Covers any non-standard response
   
    service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        data[col] = data[col].apply(encode_service)
   
    # Encode Contract: Month-to-month -> 0, One year -> 1, Two year -> 2
    data['Contract'] = data['Contract'].apply(lambda x: 0 if x=="Month-to-month" else 1 if x=="One year" else 2)
   
    # Encode PaymentMethod: Electronic check -> 0, Mailed check -> 1, Bank transfer (automatic) -> 2, Credit card (automatic) -> 3
    data['PaymentMethod'] = data['PaymentMethod'].apply(
        lambda x: 0 if x=="Electronic check" else 1 if x=="Mailed check"
        else 2 if x=="Bank transfer (automatic)" else 3
    )
   
    # Ensure target variable (Churn) is numeric
    if data['Churn'].dtype == 'object':
        data['Churn'] = data['Churn'].apply(lambda x: 1 if x=="Yes" else 0)
   
    return data

df_processed = preprocess_data(df)

# Train-Test Split and Model Training
def train_model(data):
    features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    X = data[features]
    y = data['Churn']
   
    # Split the data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model selection
    #model_choice = st.selectbox("Choose Model", ["Random Forest", "Gradient Boosting"])
    #if model_choice == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    #else:
       # model = GradientBoostingClassifier(n_estimators=100, random_state=42)
   
    model.fit(X_train, y_train)
   
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
   
    # Get feature importance
    feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
   
    return model, acc, cm, feature_importance, X_train, X_test, y_train, y_test

model, acc, cm, feature_importance, X_train, X_test, y_train, y_test = train_model(df_processed)

# Define the function properly before calling it
def visualize_data(df):
   plt.style.use('dark_background')

   # Customer Demographics Section
   st.subheader("📊 Customer Demographics")

   col1, col2 = st.columns(2)

   with col1:
       st.write("### Tenure Distribution")
       fig, ax = plt.subplots()
       sns.histplot(df['tenure'], ax=ax, color='cyan', bins=30)
       ax.set_xlabel("Tenure (Months)")
       ax.set_ylabel("Count")
       st.pyplot(fig)

   with col2:
       st.write("### Monthly Charges by Churn")
       fig, ax = plt.subplots()
       sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=ax, palette="coolwarm")
       ax.set_xlabel("Churn (0 = No, 1 = Yes)")
       ax.set_ylabel("Monthly Charges ($)")
       st.pyplot(fig)

   # Service Usage Patterns
   st.subheader("📡 Service Usage Patterns")
   st.write("### Monthly Charges vs. Total Charges")

   fig, ax = plt.subplots()
   sns.scatterplot(x='MonthlyCharges', y='TotalCharges', hue='Churn', data=df, ax=ax, palette="coolwarm", alpha=0.7)
   ax.set_xlabel("Monthly Charges ($)")
   ax.set_ylabel("Total Charges ($)")
   ax.set_title("Monthly Charges vs. Total Charges (Colored by Churn)")
   st.pyplot(fig)


# Sidebar Navigation
st.sidebar.title("Navigation")
#Sidebar options for navigation
option = st.sidebar.radio("🔍 Choose Analysis", 
    ["📊 Data Insights Hub","🎨Smart Data Visuals", "🤖 Churn Explore & Predict", "📈 Model Evaluation"])
#option = st.sidebar.radio("Go to", ["Customer Profile Analysis", "Visualizations", "Churn Risk Assessment","🔍 Interactive Data Exploration", "Churn Prediction"])

# Page1 Customer Profile Analysis
# Page 1: Customer Profile Analysis
if option == "📊 Data Insights Hub":
    # Add custom CSS for new elements
    st.markdown("""
        <style>
            .upload-box {
                border: 2px dashed #4b9ac9;
                border-radius: 15px;
                padding: 20px;
                margin: 20px 0;
                text-align: center;
                background: rgba(75, 154, 201, 0.05);
            }
            .quote-card {
                background: linear-gradient(145deg, #2a3f5f, #4b9ac9);
                border-radius: 15px;
                padding: 25px;
                margin: 20px 0;
                color: white;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .stats-table {
                background: #ffffff;
                border-radius: 15px;
                padding: 15px;
                margin: 15px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            /* Compact Filter Container */
            .compact-filter {
                background: rgba(255,255,255,0.95);
                border-radius: 12px;
                padding: 15px;
                margin: 10px 0;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                border: 1px solid #e0e0e0;
            }
        </style>
    """, unsafe_allow_html=True)

    # Add File Uploader
    # Create Button for File Upload
    if st.button("📤 Upload Custom Dataset"):
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
          try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
          except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

    # **Ensure df exists before using it**
    if 'df' in locals():
        # **🔹 Filter Controls (Properly Defined Widgets)**
        st.markdown('<div class="compact-filter">', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            tenure = st.slider("📅 Tenure (Months)", 0, 72, (0, 72), key="tenure")

        with col2:
            charges = st.slider("💰 Monthly Charges", 0, 120, (0, 120), key="charges")

        with col3:
            contracts = st.multiselect("📑 Contracts", df['Contract'].unique(), df['Contract'].unique(), key="contracts")

        with col4:
            payments = st.multiselect("💳 Payments", df['PaymentMethod'].unique(), df['PaymentMethod'].unique(), key="payments")

        st.markdown('</div>', unsafe_allow_html=True)
    
    # **🔹 Apply Filters to Data**
    filtered_df = df[
            (df['tenure'] >= tenure[0]) & 
            (df['tenure'] <= tenure[1]) & 
            (df['MonthlyCharges'] >= charges[0]) & 
            (df['MonthlyCharges'] <= charges[1]) & 
            (df['Contract'].isin(contracts)) & 
            (df['PaymentMethod'].isin(payments))
        ]

    # Enhanced Data Display
    st.subheader("📋 Filtered Data Analysis")
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("""
        <div class="stats-table">
            <h4 style='color:#2a3f5f; margin:0 0 10px 0;'>Numerical Summary</h4>
        """, unsafe_allow_html=True)
        numerical_stats = filtered_df.describe().T.drop('Churn', errors='ignore')
        st.dataframe(numerical_stats.style.format("{:.2f}"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="stats-table">
            <h4 style='color:#2a3f5f; margin:0 0 10px 0;'>Filtered Data Preview</h4>
        """, unsafe_allow_html=True)
        st.dataframe(filtered_df, height=400, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Enhanced Download Section
    st.markdown("---")
    st.subheader("📥 Data Export")
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        st.download_button(
            label="Download Filtered Data (CSV)",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name='filtered_telco_data.csv',
            mime='text/csv')
    with export_col2:
        st.download_button(
            label="Download Full Dataset (CSV)",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='full_telco_data.csv',
            mime='text/csv'
        )          

###############page2 Smart Data Visuals
elif option == "🎨Smart Data Visuals":
    # ... [Keep existing CSS] ...

    # Custom CSS for metrics and buttons
    st.markdown("""
    <style>
        div[data-testid="metric-container"] {
            background: linear-gradient(145deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border: 1px solid #dee2e6;
            transition: transform 0.2s;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
        }
        div[data-testid="stHorizontalBlock"] > div[role="radiogroup"] {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        div[role="radiogroup"] label {
            padding: 10px 20px !important;
            border-radius: 8px !important;
            transition: all 0.3s !important;
        }
        div[role="radiogroup"] label:hover {
            background: #e9ecef !important;
        }
        div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {
            background: #4b9ac9 !important;
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Metrics Section
    st.subheader("📊 Real-time Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    df_processed['CLV'] = df_processed['MonthlyCharges'] * df_processed['tenure']
    
    with col1:
        st.metric("Total Customers", len(df_processed), 
                 help="Number of customers in current selection")

    with col2:
        avg_tenure = df_processed['tenure'].mean()
        st.metric("Avg Tenure", f"{avg_tenure:.1f} mo", 
                 delta=f"{avg_tenure - df['tenure'].mean():.1f} vs global")

    with col3:
        churn_rate = (df_processed['Churn'].value_counts(normalize=True).get(1, 0)) * 100
        global_churn_rate = df['Churn'].value_counts(normalize=True).get(1, 0) * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%", 
                 delta_color="inverse", 
                 delta=f"{churn_rate - global_churn_rate:.1f}% vs global")

    with col4:
        avg_clv = df_processed['CLV'].mean()
        st.metric("Avg CLV", f"${avg_clv:,.0f}", 
                 delta=f"${avg_clv - (df['MonthlyCharges'] * df['tenure']).mean():.0f} vs global")

    # Charts Section
    with st.container():
        st.subheader("📈 Interactive Charts Explorer")
        chart_choice = st.radio("Select Chart Type:", 
                               ["Gender Distribution", "Tenure Distribution", "Service Analysis","Customer Composition"],
                               horizontal=True,
                               key="main_charts",
                               label_visibility="collapsed")
        
        chart_container = st.container()
        with chart_container:
            if chart_choice == "Gender Distribution":
                fig = px.pie(df_processed, names='gender', 
                            title="<b>Gender Distribution</b>",
                            color_discrete_sequence=['#4b9ac9', '#ff6b6b'],
                            hole=0.4)
                fig.update_layout(showlegend=True, height=400, 
                                margin=dict(t=60, b=20), 
                                font=dict(size=12))
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart_choice == "Tenure Distribution":
                fig = px.histogram(df_processed, x='tenure', nbins=20, 
                                 title='<b>Tenure Distribution</b>',
                                 color_discrete_sequence=['#4b9ac9'])
                fig.update_layout(height=400, bargap=0.1,
                                xaxis_title='Tenure (months)',
                                yaxis_title='Number of Customers',
                                margin=dict(t=60, b=40))
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart_choice == "Service Analysis":
                col1, col2 = st.columns([1, 3])
                with col1:
                    service_cols = ['PhoneService', 'InternetService', 'StreamingTV', 'TechSupport']
                    selected_service = st.selectbox(
                        "Select Service to Analyze", 
                        service_cols,
                        format_func=lambda x: x.replace('Service', ''),
                        key='service_selector'
                    )
                with col2:
                    fig = px.bar(df_processed, 
                                x=selected_service, 
                                color='Churn',
                                barmode='group',
                                title=f'<b>{selected_service.replace("Service", "")} Adoption vs Churn</b>',
                                color_discrete_map={'Yes': '#ff6b6b', 'No': '#4b9ac9'},
                                height=400)
                    fig.update_layout(
                        xaxis_title='Service Status',
                        yaxis_title='Number of Customers',
                        margin=dict(t=60, b=40),
                        plot_bgcolor='rgba(0,0,0,0.02)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            else:
                col1, col2 = st.columns([1, 3])
                with col1:
                    sunburst_cols = st.multiselect("Select dimensions:", 
                                                  ['Contract', 'PaymentMethod', 'InternetService', 'gender'],
                                                  default=['Contract', 'PaymentMethod'],
                                                  key='sunburst_dims')
                with col2:
                    if sunburst_cols:
                        fig = px.sunburst(df_processed, path=sunburst_cols + ['Churn'], 
                                        color='Churn', 
                                        color_discrete_map={'Yes': '#ff6b6b', 'No': '#4b9ac9'},
                                        height=500)
                        fig.update_layout(margin=dict(t=30, l=0, r=0, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Select dimensions to view composition")

    st.divider()

    # Customer Value Insights
    st.subheader("📈 Customer Value Insights")
    tab1, tab2, tab3 = st.tabs(["CLV Analysis", "Churn Patterns", "Customer Segmentation"])

    with tab1:
        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown("#### Group by:")
            color_by = st.selectbox("", ['None', 'Contract', 'PaymentMethod', 'InternetService'], 
                                  key='clv_color')
        with col2:
            color_argument = None if color_by == 'None' else color_by
            df_processed['CLV'] = df_processed['MonthlyCharges'] * df_processed['tenure']
            
            fig = px.histogram(df_processed, x='CLV', nbins=20,
                             color=color_argument, 
                             marginal="box",
                             template='plotly_white',
                             color_discrete_sequence=px.colors.qualitative.Pastel,
                             height=450)
            
            fig.update_layout(title="Customer Lifetime Value Distribution",
                            xaxis_title="CLV ($)",
                            yaxis_title="Count",
                            margin=dict(l=40, r=40, t=60, b=40))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.pie(df_processed, names='Churn', 
                   title='Churn Distribution',
                   color_discrete_sequence=['#4b9ac9', '#ff6b6b'],
                   height=450)
        fig.update_layout(margin=dict(t=60, b=40))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce').fillna(0)
        
        fig = px.scatter_3d(df_processed,
                          x='tenure',
                          y='MonthlyCharges',
                          z='CLV',
                          color='Contract',
                          size='TotalCharges',
                          opacity=0.8,
                          color_discrete_sequence=px.colors.qualitative.Pastel,
                          height=500)
        
        fig.update_layout(title="Customer Segmentation 3D Analysis",
                        margin=dict(l=40, r=40, t=60, b=40),
                        scene=dict(
                            xaxis_title='Tenure (months)',
                            yaxis_title='Monthly Charges ($)',
                            zaxis_title='CLV ($)'
                        ))
        st.plotly_chart(fig, use_container_width=True)                                                              
#Page3  Churn Prediction
elif option == "🤖 Churn Explore & Predict":
    # st.image("Churn.png", use_container_width=True)
    st.write("## 🔮 Churn Prediction")

# Churn Quotes Section
    st.markdown("""
    <div class="quote-card">
     <blockquote style='font-style: italic; margin: 5px 0;'>
     "Acquiring a new customer costs 5x more than retaining an existing one."
     </blockquote>
     <blockquote style='font-style: italic; margin: 5px 0;'>
     "A 5% increase in customer retention can increase profits by 25-95%."
     </blockquote>
     </div>
""", unsafe_allow_html=True)    
    # Apply custom styling to enhance visibility on black background
    st.markdown("""
    <style>
        label {
            color: #ff00ff !important;  /* Neon Pink */
            font-weight: bold;
            font-size: 20px;
        }
    </style>
    """, unsafe_allow_html=True)
    # User input for features
    tenure = st.number_input("Tenure (Months)", min_value=0, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=1500.0)

    # Categorical inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method", 
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

    # Encode inputs consistently with training
    gender_encoded = 1 if gender == "Male" else 0
    partner_encoded = 1 if partner == "Yes" else 0
    dependents_encoded = 1 if dependents == "Yes" else 0
    phone_service_encoded = 1 if phone_service == "Yes" else 0

    def encode_service_input(x):
        if x == "Yes":
            return 2
        elif x == "No":
            return 0
        else:
            return 1

    multiple_lines_encoded = encode_service_input(multiple_lines)
    online_security_encoded = encode_service_input(online_security)
    online_backup_encoded = encode_service_input(online_backup)
    device_protection_encoded = encode_service_input(device_protection)
    tech_support_encoded = encode_service_input(tech_support)
    streaming_tv_encoded = encode_service_input(streaming_tv)
    streaming_movies_encoded = encode_service_input(streaming_movies)

    contract_encoded = 0 if contract == "Month-to-month" else 1 if contract == "One year" else 2
    paperless_billing_encoded = 1 if paperless_billing == "Yes" else 0
    payment_method_encoded = (
        0 if payment_method == "Electronic check" 
        else 1 if payment_method == "Mailed check" 
        else 2 if payment_method == "Bank transfer (automatic)" 
        else 3
    )

    # Create input data DataFrame
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'gender': [gender_encoded],
        'Partner': [partner_encoded],
        'Dependents': [dependents_encoded],
        'PhoneService': [phone_service_encoded],
        'MultipleLines': [multiple_lines_encoded],
        'OnlineSecurity': [online_security_encoded],
        'OnlineBackup': [online_backup_encoded],
        'DeviceProtection': [device_protection_encoded],
        'TechSupport': [tech_support_encoded],
        'StreamingTV': [streaming_tv_encoded],
        'StreamingMovies': [streaming_movies_encoded],
        'Contract': [contract_encoded],
        'PaperlessBilling': [paperless_billing_encoded],
        'PaymentMethod': [payment_method_encoded]
    })

    if st.button("Predict Churn"):
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][prediction]
        result = "🔥 Customer is likely to churn!" if prediction == 1 else "✅ Customer is likely to stay!"
        st.subheader("Prediction Result")
        st.write(f"Confidence: {prob*100:.2f}%")
        st.markdown(f"<h3 style='color:{'#f0f' if prediction == 1 else '#0ff'}'>{result}</h3>", unsafe_allow_html=True)

#Page 4 Model Evaluation
elif option == "📈 Model Evaluation":
    st.write("## 📈 Model Evaluation Metrics")
    
    # Neon-styled compact accuracy display
    st.markdown(f"""
    <style>
        .neon-box {{
            background: linear-gradient(145deg, #8A2BE2, #00FFFF);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
            margin: 1rem auto;
            max-width: 400px;
            border: 1px solid #00FFFF;
        }}
        .neon-heading {{
            color: white;
            margin: 0 0 0.5rem 0;
            font-size: 1.8rem;
            font-family: 'Courier New', monospace;
            text-shadow: 0 0 10px rgba(255,255,255,0.5);
        }}
        .neon-value {{
            font-size: 2.5rem;
            color: white;
            font-weight: 900;
            margin: 0;
            letter-spacing: 2px;
            text-shadow: 0 0 15px #8A2BE2;
        }}
    </style>
    
    <div class="neon-box">
        <div class="neon-heading">🚀 Model Accuracy</div>
        <div class="neon-value">{acc*100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
    # Create two columns for horizontal layout
    col1, col2 = st.columns(2)

    # Confusion Matrix (Left Column)
    with col1:
        st.write("### 🧮 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                   linewidths=.5, annot_kws={'size': 14}, ax=ax)
        plt.title('Classification Confusion Matrix', fontsize=14, pad=20)
        st.pyplot(fig)

    # Feature Importance (Right Column)
    with col2:
        st.write("### 🔍 Feature Importance")
        fig, ax = plt.subplots(figsize=(8, 6))
        feature_importance.plot(kind='barh', ax=ax, 
                               color=sns.color_palette('viridis', len(feature_importance)))
        plt.title('Feature Importance Ranking', fontsize=14, pad=20)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        st.pyplot(fig)

    # Correlation Matrix (Full Width Below)
    st.write("### 🔗 Correlation Matrix")
    numeric_df = df_processed.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', 
               cmap='coolwarm', center=0, linewidths=.5, ax=ax)
    plt.title('Feature Correlation Matrix', fontsize=14, pad=20)
    st.pyplot(fig)
    
