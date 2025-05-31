from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
import os
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import logging
from pydantic import BaseModel
import uvicorn
import sys
from tensorflow.keras.models import load_model

# Keep your existing import as is:

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(
    title="BioLens Gene Expression API",
    description="ML-powered gene expression analysis for disease prediction",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001","http://localhost:5173"],  # Add your React app URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import os
import joblib
import logging
import tensorflow 
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class PredictionResult(BaseModel):
    sample_name: str
    predicted_class: str
    cancer_probability: float
    confidence_score: float
    top_genes_expression: Dict[str, float]
    
class VisualizationData(BaseModel):
    bar_chart_data: Dict[str, float]
    line_chart_data: Dict[str, float]
    expression_summary: Dict[str, Any]

class BioLensResponse(BaseModel):
    prediction: PredictionResult
    visualizations: VisualizationData
    processing_info: Dict[str, Any]
    success: bool
    message: str
    
class ModelArtifacts:
    """Singleton class to load and cache model artifacts"""
    _instance = None
    _artifacts_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelArtifacts, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._artifacts_loaded:
            self.load_artifacts()
            self._artifacts_loaded = True
    
    def load_artifacts(self):
        """Load all model artifacts"""
        self.selected_genes = None
        self.scaler = None
        self.sklearn_model = None
        self.keras_model = None
        self.ensemble_weights = None
        
        
        if os.path.exists('selected_genes.pkl'):
            try:
                self.selected_genes = joblib.load('selected_genes.pkl')
                logger.info(f"Loaded {len(self.selected_genes)} selected genes")
            except Exception as e:
                logger.error(f"Failed to load selected_genes.pkl: {e}")
        
        
        if os.path.exists('scaler.pkl'):
            try:
                self.scaler = joblib.load('scaler.pkl')
                logger.info("Loaded scaler successfully")
            except Exception as e:
                logger.error(f"Failed to load scaler.pkl: {e}")
        
   
        if os.path.exists('sk_model.pkl'):
            try:
                self.sklearn_model = joblib.load('sk_model.pkl')
                logger.info("Loaded sklearn model successfully")
            except Exception as e:
                logger.error(f"Failed to load sk_model.pkl: {e}")
        
        
        if os.path.exists('nn_model.h5'):
            try:
                from tensorflow.keras.models import load_model
                self.keras_model = load_model('nn_model.h5')
                logger.info("Loaded Keras model successfully")
            except Exception as e:
                logger.error(f"Failed to load nn_model.h5: {e}")
        
       
        if os.path.exists('ensemble_weights.pkl'):
            try:
                self.ensemble_weights = joblib.load('ensemble_weights.pkl')
                logger.info("Loaded ensemble weights")
            except Exception as e:
                logger.error(f"Failed to load ensemble weights: {e}")
                
                self.ensemble_weights = {'sklearn': 0.6, 'keras': 0.4}
    
    def validate_artifacts(self):
        """Check if all required artifacts are loaded"""
        missing = []
        if self.selected_genes is None:
            missing.append("selected_genes")
        if self.scaler is None:
            missing.append("scaler")
        if self.sklearn_model is None and self.keras_model is None:
            missing.append("at least one model (sklearn or keras)")
        
        return len(missing) == 0, missing

class GeneExpressionProcessor:
    """Handle gene expression data processing and prediction"""
    
    def __init__(self, artifacts: ModelArtifacts):
        self.artifacts = artifacts
    
    def preprocess_data(self, df: pd.DataFrame) -> tuple:
        """Preprocess uploaded TSV data"""
        try:
            # Log2 transformation
            df_log2 = np.log2(df + 1)
            
            # Check for missing genes
            missing_genes = [gene for gene in self.artifacts.selected_genes 
                           if gene not in df_log2.index]
            
            if missing_genes:
                raise ValueError(f"Missing {len(missing_genes)} required genes: {missing_genes[:5]}...")
            
            # Extract features for prediction
            features = df_log2.loc[self.artifacts.selected_genes].T
            data_scaled = self.artifacts.scaler.transform(features)
            
            return df_log2, data_scaled, None
            
        except Exception as e:
            return None, None, str(e)
    
    def predict_with_ensemble(self, data_scaled: np.ndarray) -> tuple:
        """Make predictions using ensemble of models"""
        predictions = {}
        
        # Sklearn prediction
        if self.artifacts.sklearn_model is not None:
            try:
                if hasattr(self.artifacts.sklearn_model, 'predict_proba'):
                    prob_sklearn = self.artifacts.sklearn_model.predict_proba(data_scaled)[0, 1]
                else:
                    prob_sklearn = self.artifacts.sklearn_model.predict(data_scaled)[0]
                predictions['sklearn'] = float(prob_sklearn)
            except Exception as e:
                logger.error(f"Sklearn prediction failed: {e}")
        
        # Keras prediction
        if self.artifacts.keras_model is not None:
            try:
                prob_keras = float(self.artifacts.keras_model.predict(data_scaled, verbose=0).flatten()[0])
                predictions['keras'] = prob_keras
            except Exception as e:
                logger.error(f"Keras prediction failed: {e}")
        
        if not predictions:
            raise ValueError("No models available for prediction")
        
        # Ensemble prediction
        if len(predictions) > 1 and self.artifacts.ensemble_weights:
            ensemble_prob = sum(predictions[model] * self.artifacts.ensemble_weights.get(model, 0) 
                             for model in predictions)
            ensemble_prob = np.clip(ensemble_prob, 0.0, 1.0)
        else:
            # Use single model or average if no weights
            ensemble_prob = np.mean(list(predictions.values()))
        
        return ensemble_prob, predictions
    
    def generate_visualizations(self, df_log2: pd.DataFrame, sample_name: str) -> Dict:
        """Generate visualization data for frontend"""
        try:
            # Top 5 genes expression
            top5_genes = self.artifacts.selected_genes[:5]
            top5_data = df_log2.loc[top5_genes, sample_name].to_dict()
            
            # All selected genes for line chart
            all_genes_data = df_log2.loc[self.artifacts.selected_genes, sample_name].to_dict()
            
            # Expression summary statistics
            expression_stats = {
                'mean_expression': float(df_log2[sample_name].mean()),
                'median_expression': float(df_log2[sample_name].median()),
                'std_expression': float(df_log2[sample_name].std()),
                'min_expression': float(df_log2[sample_name].min()),
                'max_expression': float(df_log2[sample_name].max()),
                'total_genes': len(df_log2),
                'selected_genes_count': len(self.artifacts.selected_genes)
            }
            
            return {
                'bar_chart_data': top5_data,
                'line_chart_data': all_genes_data,
                'expression_summary': expression_stats
            }
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            return {
                'bar_chart_data': {},
                'line_chart_data': {},
                'expression_summary': {}
            }



@app.get('/')
async def root():
    """Health check endpoint"""
    artifacts_valid, missing = model_artifacts.validate_artifacts()
    return {
        "message": "BioLens Gene Expression API",
        "status": "healthy" if artifacts_valid else "missing artifacts",
        "missing_artifacts": missing,
        "version": "1.0.0"
    }
@app.post("/predict", response_model=BioLensResponse)
async def predict_gene_expression(file: UploadFile = File(...)):
    """
    Main prediction endpoint
    Upload a TSV file with gene expression data and get predictions + visualizations
    """
    # Validate artifacts
    artifacts_valid, missing = model_artifacts.validate_artifacts()
    if not artifacts_valid:
        raise HTTPException(
            status_code=503, 
            detail=f"Service unavailable. Missing artifacts: {missing}"
        )
    
    # Validate file
    if not file.filename.lower().endswith('.tsv'):
        raise HTTPException(
            status_code=400, 
            detail="File must be a TSV (.tsv) file"
        )
    
    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')), sep='\t', index_col=0)
        
        if df.empty or df.shape[1] == 0:
            raise HTTPException(status_code=400, detail="Empty or invalid TSV file")
        
        sample_name = df.columns[0]
        logger.info(f"Processing sample: {sample_name}")
        
        # Preprocess data
        df_log2, data_scaled, error = processor.preprocess_data(df)
        if error:
            raise HTTPException(status_code=400, detail=f"Preprocessing failed: {error}")
        
        # Make predictions
        cancer_prob, individual_predictions = processor.predict_with_ensemble(data_scaled)
        
        # Determine class and confidence
        predicted_class = 'Cancer' if cancer_prob > 0.5 else 'Normal'
        confidence_score = max(cancer_prob, 1 - cancer_prob)  # Distance from 0.5
        
        # Get top genes expression
        top_genes = model_artifacts.selected_genes[:5]
        top_genes_expr = df_log2.loc[top_genes, sample_name].to_dict()
        
        # Generate visualizations
        viz_data = processor.generate_visualizations(df_log2, sample_name)
        
        # Prepare response
        response = BioLensResponse(
            prediction=PredictionResult(
                sample_name=sample_name,
                predicted_class=predicted_class,
                cancer_probability=round(float(cancer_prob), 6),
                confidence_score=round(float(confidence_score), 4),
                top_genes_expression=top_genes_expr
            ),
            visualizations=VisualizationData(**viz_data),
            processing_info={
                'total_genes_in_sample': df.shape[0],
                'selected_genes_used': len(model_artifacts.selected_genes),
                'models_used': list(individual_predictions.keys()),
                'individual_predictions': individual_predictions,
                'ensemble_weights': model_artifacts.ensemble_weights
            },
            success=True,
            message="Prediction completed successfully"
        )
        
        logger.info(f"Prediction successful for {sample_name}: {predicted_class} ({cancer_prob:.4f})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)