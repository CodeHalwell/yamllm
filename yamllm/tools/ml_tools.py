from .base import Tool
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, Any

class DataLoader(Tool):
    def __init__(self):
        super().__init__(
            name="data_loader",
            description="Load data from various file formats"
        )
        self.supported_formats = ['.csv', '.xlsx', '.json', '.parquet']

    def execute(self, filepath: str, **kwargs) -> pd.DataFrame:
        ext = filepath.lower().split('.')[-1]
        if ext == 'csv':
            return pd.read_csv(filepath, **kwargs)
        elif ext == 'xlsx':
            return pd.read_excel(filepath, **kwargs)
        elif ext == 'json':
            return pd.read_json(filepath, **kwargs)
        elif ext == 'parquet':
            return pd.read_parquet(filepath, **kwargs)
        raise ValueError(f"Unsupported file format: {ext}")

class EDAAnalyzer(Tool):
    def __init__(self):
        super().__init__(
            name="eda_analyzer",
            description="Perform exploratory data analysis"
        )

    def execute(self, data: pd.DataFrame) -> Dict[str, Any]:
        analysis = {
            'basic_info': {
                'shape': data.shape,
                'dtypes': data.dtypes.to_dict(),
                'missing_values': data.isnull().sum().to_dict(),
                'duplicates': data.duplicated().sum()
            },
            'numerical_summary': data.describe().to_dict(),
            'categorical_summary': {
                col: data[col].value_counts().to_dict() 
                for col in data.select_dtypes(include=['object']).columns
            }
        }
        return analysis

class DataPreprocessor(Tool):
    def __init__(self):
        super().__init__(
            name="data_preprocessor",
            description="Clean and preprocess data"
        )
        self.scalers = {}
        self.encoders = {}

    def execute(self, 
                data: pd.DataFrame,
                numeric_strategy: str = 'mean',
                categorical_strategy: str = 'mode',
                scale: bool = True) -> pd.DataFrame:
        df = data.copy()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            if numeric_strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif numeric_strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
                
        for col in categorical_cols:
            if categorical_strategy == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
                
        # Scale numeric features
        if scale:
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self.scalers['standard'] = scaler
            
        # Encode categorical features
        for col in categorical_cols:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            self.encoders[col] = encoder
            
        return df

class ModelTrainer(Tool):
    def __init__(self):
        super().__init__(
            name="model_trainer",
            description="Train and evaluate ML models"
        )

    def execute(self, 
                data: pd.DataFrame,
                target: str,
                model,
                test_size: float = 0.2,
                random_state: int = 42) -> Dict[str, Any]:
        # Split features and target
        X = data.drop(columns=[target])
        y = data[target]
        
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        results = {
            'train_score': model.score(X_train, y_train),
            'test_score': model.score(X_test, y_test),
            'model': model,
            'predictions': model.predict(X_test),
            'feature_importance': None
        }
        
        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            results['feature_importance'] = dict(
                zip(X.columns, model.feature_importances_)
            )
            
        return results

class ModelEvaluator(Tool):
    def __init__(self):
        super().__init__(
            name="model_evaluator",
            description="Evaluate model performance with various metrics"
        )

    def execute(self, 
                y_true: np.ndarray,
                y_pred: np.ndarray,
                task_type: str = 'classification') -> Dict[str, float]:
        from sklearn.metrics import (accuracy_score, precision_score,
                                   recall_score, f1_score, r2_score,
                                   mean_squared_error, mean_absolute_error)
        
        if task_type == 'classification':
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted')
            }
        else:  # regression
            return {
                'r2': r2_score(y_true, y_pred),
                'mse': mean_squared_error(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
            }