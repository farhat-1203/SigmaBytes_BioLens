# 🔬 BioLens: A Gene Expression Analysis Tool for Disease Diagnosis


## 📌 Project Overview

**Team Name:** Sigma Bytes  
**Project Title:** BioLens: A Gene Expression Analysis Tool for Disease Diagnosis

Analyzing high-dimensional gene expression data to identify disease-linked genes is complex, time-consuming, and often inaccessible to non-experts. **BioLens** aims to democratize access to bioinformatics by providing a simple, intuitive interface that enables researchers and clinicians to explore, analyze, and visualize gene expression datasets from the NCBI GEO database.

---

## 💡 Problem Statement

Identifying disease-associated genes from large-scale gene expression data can be overwhelming, especially for non-coders or early-stage biomedical researchers. There’s a pressing need for tools that simplify differential gene expression (DGE) analysis and result interpretation.

---

## 🚀 Proposed Solution

**BioLens** is a web-based tool that enables users to:
- Upload and analyze gene expression datasets from GEO
- Perform DGE analysis using robust statistical methods
- Visualize results through interactive plots like Volcano plots and UMAP
- Optionally predict disease status using machine learning models

---

## 🔍 Dataset Used

- **Dataset:** [GSE50760](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE50760) from NCBI GEO  
- **Description:** Includes normal, primary colorectal cancer, and metastatic samples.  
- **Analysis Focus:** Differential expression between normal and colorectal cancer (CRC) samples.

---

## Technical Architecture

### Dataset Pipeline
- Public microarray dataset (GSE50760) in `.tsv` format
- Preprocessed and normalized gene expression levels (~20K features)
- Sample-wise classification based on expression profiles

### ML Model
- Feature selection using **Artificial Neural Network (ANN)**
- Classification using **RandomForestClassifier**
- Output: `Normal` or `Cancer` with associated confidence score

### TECH STACK:
- pandas, numpy, sklearn
- seaborn, matplotlib, plotly
- TensorFlow, Keras
- Streamlit (for frontend & deployment)

---

## 📊 Key Features

- **Differential Gene Expression Analysis** (DESeq2)
- **UMAP Plot** for dimensionality reduction and clustering
- **Volcano Plot** for significant DEGs
- **Dispersion Estimation** for quality control
---

## 📈 Visual Results

### 🔹 UMAP Plot  
<img width="360" alt="image" src="https://github.com/user-attachments/assets/20924b05-d8ef-47d7-848d-f7a9cd0890f5" />


### 🔹 Volcano Plot  
<img width="341" alt="image" src="https://github.com/user-attachments/assets/432f5632-175e-415a-9191-4148028b80c2" />


### 🔹 Dispersion Plot  
<img width="378" alt="image" src="https://github.com/user-attachments/assets/2eaa8325-669a-4c9c-b95c-6b4d2030a13d" />

---

## 🧪 Implementation Details

- Parsed and filtered gene expression matrix from GSE50760
- Used DESeq2 to normalize counts and identify DEGs (adj. p-value < 0.05)
- Generated visualizations for interpretation of DEGs
- Validated findings with known literature

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/farhat-1203/SigmaBytes_BioLens.git
cd BioLens
```

### 2. Create a Virtual Environment 
```
bash
# For Windows
python -m venv env
env\Scripts\activate
```

### 3. Install the Dependencies
```
bash
pip install -r requirements.txt
```

### 4. Start the Streamlit application:
```
bash
streamlit run app.py
```
