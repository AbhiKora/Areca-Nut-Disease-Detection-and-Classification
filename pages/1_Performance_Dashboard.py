# pages/1_📊_Performance_Dashboard.py
import streamlit as st
import numpy as np
import os
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Performance Dashboard",
    layout="wide"
)

# --- App Title ---
st.title("📊 Model Performance Dashboard")
st.write("An overview of the performance metrics and visualizations for each trained model.")

# --- Data Loading ---
@st.cache_data
def load_results(filepath):
    """Loads the model results dictionary from the .npy file."""
    if os.path.exists(filepath):
        return np.load(filepath, allow_pickle=True).item()
    return None

results = load_results('model_results.npy')

# --- Main Application ---
if results is None:
    st.error(
        "**Error:** `model_results.npy` not found. "
        "Please ensure the file is in the root directory of your project."
    )
else:
    model_names = list(results.keys())
    
    selected_model = st.selectbox(
        'Choose a model to see its performance details:',
        model_names
    )

    st.header(f'Performance Analysis for {selected_model}')
    
    # --- Display Metrics in a Table ---
    st.subheader('📋 Performance Metrics')
    metrics = results[selected_model]
    metrics_df = pd.DataFrame([metrics])
    st.table(metrics_df.style.format("{:.4f}"))
    
    # --- Display Plots in Columns ---
    st.subheader('📈 Visualizations')
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Confusion Matrix")
        cm_path = f'{selected_model}_confusion_matrix.png'
        if os.path.exists(cm_path):
            st.image(cm_path, caption=f'Confusion Matrix for {selected_model}', use_container_width=True)
        else:
            st.warning(f"Plot not found: `{cm_path}`")

    with col2:
        st.markdown("#### t-SNE Plot of Image Features")
        tsne_path = f'{selected_model}_tsne_plot.png'
        if os.path.exists(tsne_path):
            st.image(tsne_path, caption=f't-SNE Visualization for {selected_model}', use_container_width=True)
        else:
            st.warning(f"Plot not found: `{tsne_path}`")