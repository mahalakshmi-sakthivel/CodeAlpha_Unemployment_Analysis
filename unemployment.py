import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress all background system and version warnings
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(page_title="Unemployment Analysis Dashboard", page_icon="📉", layout="wide")

st.title("📉 Unemployment Analysis Dashboard")
st.markdown("Developed for the **CodeAlpha Data Science Internship**. Built using **Pandas, Seaborn, and Matplotlib**.")

try:
    # 1. Load the specific, cleaner dataset file from your extracted folder
    df = pd.read_csv('Unemployment_Rate_upto_11_2020.csv')
    
    # Standardize column names to remove hidden trailing spaces and dots
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('.', '')
    
    st.subheader("📋 1. Unemployment Dataset Preview (Pandas DataFrame)")
    st.dataframe(df.head(), use_container_width=True)
    
    # --- AUTOMATIC COLUMN MAPPING VIA DATA TYPE CHARACTERISTICS ---
    # Convert all column string mappings to lowercase to handle casing bugs safely
    col_mapping = {col.lower(): col for col in df.columns}
    
    # Match the core columns dynamically
    date_col = col_mapping.get('date', None)
    rate_col = col_mapping.get('estimated_unemployment_rate_(%)', col_mapping.get('estimated_unemployment_rate', df.select_dtypes(include=[np.number]).columns[0]))
    region_col = col_mapping.get('region', col_mapping.get('state', df.columns[0]))
    
    # Process Dates and Feature Engineering Safely if column exists
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col].str.strip(), format='%d-%m-%Y', errors='coerce')
        df['Year'] = df[date_col].dt.year
        df['Month'] = df[date_col].dt.month_name()
    
    # UI Layout Split into Left and Right Columns
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("📈 2. Unemployment Trends Over Time")
        
        fig1, ax1 = plt.subplots(figsize=(8, 4.5))
        if date_col and rate_col:
            # Group by date to get a smooth national timeline trend line
            trend_df = df.groupby(date_col)[rate_col].mean().reset_index()
            sns.lineplot(data=trend_df, x=date_col, y=rate_col, color='crimson', lw=2.5, marker='o', ax=ax1)
            plt.title('Average Unemployment Rate Trend Over Time')
            plt.xlabel('Timeline')
            plt.ylabel('Estimated Unemployment Rate (%)')
            plt.grid(True, alpha=0.3)
            st.pyplot(fig1)
        else:
            st.warning("Time-series visualization skipped due to feature detection limit.")

    with col_right:
        st.subheader("📊 3. Regional Breakdown Insights")
        
        if region_col and rate_col:
            unique_regions = sorted(df[region_col].dropna().unique().tolist())
            selected_region = st.selectbox("Select a Specific Region to Analyze:", unique_regions)
            
            region_filtered = df[df[region_col] == selected_region]
            
            fig2, ax2 = plt.subplots(figsize=(8, 4.5))
            sns.barplot(data=region_filtered, x='Year' if 'Year' in df.columns else region_col, y=rate_col, palette='viridis', ax=ax2, ci=None)
            plt.title(f'Unemployment Behavior in {selected_region}')
            plt.ylabel('Unemployment Rate (%)')
            st.pyplot(fig2)
        else:
            st.warning("Regional columns not detected for breakdown analysis.")

    st.markdown("---")
    
    # --- INTERACTIVE SHOCK ANALYZER SECTION ---
    st.subheader("🦠 4. Economic Impact Analysis Sandbox")
    st.write("Analyze the severe economic shock impact patterns across different seasons:")
    
    if 'Month' in df.columns and rate_col:
        # Sort months chronologically for clean plotting layout visualization
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        df['Month'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)
        monthly_avg = df.groupby('Month')[rate_col].mean().reset_index()
        
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        sns.barplot(data=monthly_avg, x='Month', y=rate_col, palette='magma', ax=ax3)
        plt.title('Monthly Unemployment Distribution Analysis (2020 Timeline)')
        plt.xlabel('Months')
        plt.ylabel('Average Unemployment Rate (%)')
        st.pyplot(fig3)
        
        st.info("💡 **Covid-19 Macroeconomic Insight:** Notice the unprecedented, massive spike during the mid-2020 months (April–May). This visually captures the immediate real-world global economic freeze caused by sudden lock-down parameters and industrial stalls.")

except FileNotFoundError:
    st.error("⚠️ Data Configuration Error: Please verify that the file named exactly 'Unemployment_Rate_upto_11_2020.csv' is uploaded directly inside your active repository.")
except Exception as e:
    st.error(f"🚨 Dashboard Execution Alert: {e}")