# In dashboard/dashboard_v1.py

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import requests # Used for direct MP API calls

# --- Page Configuration ---
st.set_page_config(
    page_title="Cerebro Forge Dashboard",
    page_icon="🧠",
    layout="wide" 
)

# --- Load Database Credentials ---
load_dotenv(".env")
DATABASE_URL = os.getenv("DATABASE_URL")
MP_API_KEY = os.getenv("MP_API_KEY")

if MP_API_KEY is None:
    st.error("MP_API_KEY is missing from your local .env file. Please ensure it is set.")


# --- Caching Database Connection ---
@st.cache_resource
def get_db_engine():
    """Creates and returns a SQLAlchemy engine."""
    if not DATABASE_URL:
        st.error("DATABASE_URL not found in .env file.")
        return None
    try:
        engine = create_engine(DATABASE_URL)
        return engine
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

engine = get_db_engine()


# --- Caching Data Loading (Synthetic) ---
@st.cache_data(ttl=600)
def load_data(_engine, version_id=None):
    """Loads synthetic data from the database."""
    if _engine is None:
        return pd.DataFrame()
        
    query = "SELECT formula, band_gap, formation_energy, elasticity, density, source_type, version_id FROM materials"
    if version_id and version_id != "All":
        query += f" WHERE version_id = '{version_id}'"
        
    try:
        df = pd.read_sql(query, _engine)
        return df
    except Exception as e:
        st.error(f"Error loading synthetic data: {e}")
        return pd.DataFrame()

# --- CRITICAL FIX: Function to get Real Data (Final stable method v4) ---
@st.cache_data(ttl=3600)
def load_real_data(api_key):
    """Fetches real data from Materials Project API using the basic formula endpoint."""
    if not api_key:
        return None

    try:
        # Use the older, stable /rest/v2/materials/formula/ endpoint
        BASE_URL = "https://api.materialsproject.org/rest/v2/materials/formula/" 
        
        # List of common stable formulas to query individually (fallback method)
        # We query these because they are likely to overlap with our generator's output
        common_formulas = ['Al2O3', 'MgO', 'SiO2', 'Fe2O3', 'TiO2', 'CaO', 'Na2O', 'K2O', 'ZnO', 'Cu2O', 'NiO', 'CoO', 'MnO2', 'Cr2O3', 'Si', 'Fe', 'Al', 'O2']
        
        real_materials = []
        st.sidebar.info("Querying common formulas from MP (via stable endpoint)...")
        
        for formula in common_formulas:
            try:
                # Query the core endpoint for the formula
                response = requests.get(
                    BASE_URL + formula + "/core",
                    headers={"X-API-KEY": api_key, "Content-Type": "application/json"}
                )
                response.raise_for_status() 
                data = response.json()
                
                # Extract the first valid document from the response list
                if data and 'response' in data and data['response']:
                    doc = data['response'][0]
                    
                    # Check for required properties before appending
                    if doc.get('band_gap') is not None and doc.get('formation_energy_per_atom') is not None:
                         real_materials.append({
                            "formula": doc["formula_pretty"], 
                            "band_gap_real": doc["band_gap"],
                            "formation_energy_real": doc["formation_energy_per_atom"],
                        })
            
            except requests.exceptions.HTTPError:
                # Silently skip formulas that give an error (e.g., 404)
                continue
        
        df_real = pd.DataFrame(real_materials)
        
        if df_real.empty:
            st.error("MP API Error: Failed to load any materials via the fallback formula query.")
            return None
        
        # We only use the first N unique formulas to keep the sample size manageable
        df_real = df_real.drop_duplicates(subset=['formula']).head(1000) 

        st.sidebar.success(f"Successfully loaded {len(df_real)} real materials using formula query.")
        return df_real
        
    except Exception as e:
        st.error(f"Error processing MP data: {e}")
        return pd.DataFrame()


# --- Main Dashboard Logic ---
st.title("🧠 Cerebro Forge - Synthetic Data Dashboard")
st.markdown("Foundation Phase - v1.4")

if engine:
    # --- Sidebar for Controls ---
    st.sidebar.header("Controls")
    
    try:
        versions_df = pd.read_sql("SELECT DISTINCT version_id FROM materials ORDER BY version_id DESC", engine)
        available_versions = ["All"] + versions_df['version_id'].tolist()
    except Exception:
        available_versions = ["All"]
        st.sidebar.warning("Could not fetch version IDs from DB.")

    selected_version = st.sidebar.selectbox(
        "Select Data Version:",
        options=available_versions,
        index=available_versions.index("v1.6-colab-xgboost-int") if "v1.6-colab-xgboost-int" in available_versions else 0
    )
    
    # --- Load Data ---
    df_loaded = load_data(engine, selected_version)
    df_real = load_real_data(MP_API_KEY)


    if not df_loaded.empty:
        st.header(f"📊 Data Overview (Version: {selected_version})")
        st.metric("Total Records", len(df_loaded))

        tab_stats, tab_comparison = st.tabs(["Statistics & Raw Data", "Quality Validation"])
        
        # --- Tab 1: Basic Statistics ---
        with tab_stats:
            st.subheader("Raw Data Sample")
            st.dataframe(df_loaded.head(10))

            st.subheader("Property Statistics")
            st.dataframe(df_loaded[['band_gap', 'formation_energy', 'elasticity', 'density']].describe())
        
        # --- Tab 2: Validation Comparison ---
        with tab_comparison:
            
            # Condition check: Must select a single version AND have successfully loaded real data
            if selected_version == "All" or df_real is None or df_loaded.empty:
                st.warning("Please select a single Synthetic Version (e.g., v1.6-colab-xgboost-int) and ensure the MP_API_KEY is set and working to run the comparison.")
            else:
                st.subheader(f"Validation: {selected_version} vs. Real MP Sample")

                # --- Merge DataFrames for Comparison ---
                df_synthetic = df_loaded
                df_comparison = pd.merge(df_real, df_synthetic, on="formula")
                
                if df_comparison.empty:
                    st.warning("No matching formulas found between the selected synthetic set and the real sample. Cannot perform MAE/R2 comparison.")
                else:
                    st.info(f"Comparing {len(df_comparison)} overlapping materials.")

                    # --- 1. MAE / R² Metrics ---
                    mae_fe = mean_absolute_error(df_comparison['formation_energy_real'], df_comparison['formation_energy'])
                    r2_fe = r2_score(df_comparison['formation_energy_real'], df_comparison['formation_energy'])
                    mae_bg = mean_absolute_error(df_comparison['band_gap_real'], df_comparison['band_gap'])
                    r2_bg = r2_score(df_comparison['band_gap_real'], df_comparison['band_gap'])

                    col_mae, col_r2 = st.columns(2)
                    
                    with col_mae:
                        st.metric("Formation Energy MAE (Target < 0.1 eV)", f"**{mae_fe:.4f} eV**")
                        st.metric("Band Gap MAE", f"**{mae_bg:.4f} eV**")
                        
                    with col_r2:
                        st.metric("Formation Energy R² (Target > 0.9)", f"**{r2_fe:.4f}**")
                        st.metric("Band Gap R²", f"**{r2_bg:.4f}**")
                        
                    st.markdown("---")
                    
                    # --- 2. Correlation Plots (Scatter) ---
                    st.subheader("Correlation Visuals")

                    fig_corr, axes_corr = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Formation Energy Correlation
                    sns.scatterplot(data=df_comparison, x='formation_energy_real', y='formation_energy', ax=axes_corr[0])
                    axes_corr[0].plot([-6, 2], [-6, 2], 'r--') 
                    axes_corr[0].set_title(f'FE Correlation (R²: {r2_fe:.2f})')
                    axes_corr[0].set_xlabel('Real (MP) FE (eV/atom)')
                    axes_corr[0].set_ylabel('Synthetic FE (eV/atom)')

                    # Band Gap Correlation
                    sns.scatterplot(data=df_comparison, x='band_gap_real', y='band_gap', ax=axes_corr[1])
                    axes_corr[1].plot([0, 7], [0, 7], 'r--')
                    axes_corr[1].set_title(f'BG Correlation (R²: {r2_bg:.2f})')
                    axes_corr[1].set_xlabel('Real (MP) BG (eV)')
                    axes_corr[1].set_ylabel('Synthetic BG (eV)')
                    
                    st.pyplot(fig_corr)
                    
                    # --- 3. Distribution Plots (KDE) ---
                    st.subheader("Distribution Overlap")
                    
                    fig_dist, axes_dist = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Formation Energy Distribution
                    sns.kdeplot(df_real['formation_energy_real'], label='Real (MP)', fill=True, ax=axes_dist[0], bw_adjust=0.5)
                    sns.kdeplot(df_synthetic['formation_energy'], label='Synthetic', fill=True, ax=axes_dist[0], bw_adjust=0.5)
                    axes_dist[0].set_title('Formation Energy Distribution')
                    axes_dist[0].set_xlabel('Formation Energy (eV/atom)')
                    axes_dist[0].legend()

                    # Band Gap Distribution
                    sns.kdeplot(df_real['band_gap_real'], label='Real (MP)', fill=True, ax=axes_dist[1], bw_adjust=0.5)
                    sns.kdeplot(df_synthetic['band_gap'], label='Synthetic', fill=True, ax=axes_dist[1], bw_adjust=0.5)
                    axes_dist[1].set_title('Band Gap Distribution')
                    axes_dist[1].set_xlabel('Band Gap (eV)')
                    axes_dist[1].legend()

                    st.pyplot(fig_dist)


    else:
        st.warning(f"No data loaded for version '{selected_version}'. Check database connection and data.")
else:
    st.error("Database connection failed. Cannot display dashboard.")