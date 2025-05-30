import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(page_title="Gene Expression Classifier", layout="centered")

st.title("🧬 Gene Expression Cancer Classifier")
st.write("Upload your gene expression sample (TSV) to get a cancer probability prediction and visualizations.")

# Load artifacts (selected_genes, scaler, and model)
@st.cache_resource
def load_artifacts():
    selected_genes = None
    scaler = None
    model = None
    model_type = None

    # Load selected_genes
    if os.path.exists('selected_genes.pkl'):
        try:
            selected_genes = joblib.load('selected_genes.pkl')
        except Exception:
            st.warning("Unable to load 'selected_genes.pkl'.")
    else:
        st.warning("'selected_genes.pkl' not found.")

    # Load scaler
    if os.path.exists('scaler.pkl'):
        try:
            scaler = joblib.load('scaler.pkl')
        except Exception:
            st.warning("Unable to load 'scaler.pkl'.")
    else:
        st.warning("'scaler.pkl' not found.")

    # Load sklearn model first
    if os.path.exists('sk_model.pkl'):
        try:
            model = joblib.load('sk_model.pkl')
            model_type = 'sklearn'
        except Exception:
            st.warning("Failed to load 'sk_model.pkl'.")
    # If no sklearn model, try Keras H5
    elif os.path.exists('nn_model.h5'):
        try:
            from tensorflow.keras.models import load_model
            model = load_model('nn_model.h5')
            model_type = 'keras'
        except ImportError:
            st.warning("TensorFlow/Keras not installed; cannot load 'nn_model.h5'.")
        except Exception:
            st.warning("Failed to load 'nn_model.h5'. Ensure correct format.")
    else:
        st.warning("No model file found. Place 'sk_model.pkl' or 'nn_model.h5' in the directory.")

    return selected_genes, scaler, model, model_type

selected_genes, scaler, model, model_type = load_artifacts()

# Helper function for prediction
def predict_probability(df_log2, selected_genes, scaler, model, model_type):
    features = df_log2.loc[selected_genes].T
    data_scaled = scaler.transform(features)
    try:
        if model_type == 'sklearn' and hasattr(model, 'predict_proba'):
            prob = model.predict_proba(data_scaled)[0, 1]
        elif model_type == 'keras':
            prob = float(model.predict(data_scaled).flatten()[0])
        else:
            st.error("Loaded model is not compatible.")
            return None
    except Exception:
        st.error("Error during prediction. The model may be incompatible.")
        return None
    return prob

# File uploader
uploaded_file = st.file_uploader("Choose sample TSV file", type=['tsv'])

if uploaded_file:
    # Check prerequisites
    if selected_genes is None or scaler is None or model is None:
        st.error("Missing artifacts. Cannot perform prediction.")
    else:
        try:
            df = pd.read_csv(uploaded_file, sep='\t', index_col=0)
            df_log2 = np.log2(df + 1)
        except Exception:
            st.error("Failed to read or preprocess the uploaded file.")
        else:
            sample_name = df_log2.columns[0]
            missing = [g for g in selected_genes if g not in df_log2.index]
            if missing:
                st.error(f"Missing {len(missing)} selected genes. Prediction aborted.")
            else:
                prob = predict_probability(df_log2, selected_genes, scaler, model, model_type)
                if prob is not None:
                    prob = min(max(prob + np.random.uniform(1e-6, 1e-5), 0.0), 1.0)
                    label = 'Cancer' if prob > 0.5 else 'Normal'
                    st.subheader(f"Sample: {sample_name}")
                    st.metric(label="Predicted Class", value=label)
                    st.progress(prob)
                    st.write(f"Probability of Cancer: **{prob:.6f}**")
                    st.write("### First 5 gene expression values (log2)")
                    preview = df_log2.loc[selected_genes].iloc[:5, :]
                    st.dataframe(preview)
                    if st.button("Show Visualizations"):
                        st.write("## Visualizations")
                        top5_values = df_log2.loc[selected_genes].iloc[:5, 0]
                        st.write("### Top 5 Gene Expression (log2)")
                        st.bar_chart(top5_values)
                        st.write("### All Selected Gene Expression (log2)")
                        st.line_chart(df_log2.loc[selected_genes].iloc[:, 0])
                        st.write("### Random Sample Images")
                        imgs = [np.random.rand(100,100,3) for _ in range(5)]
                        st.image(imgs, width=100, caption=[f"Image {_+1}" for _ in range(5)])

# Footer
st.write("---")
st.write("Built with Streamlit")
