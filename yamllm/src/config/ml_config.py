ML_CONFIG = {
    'data_loading': {
        'default_encoding': 'utf-8',
        'supported_formats': ['.csv', '.xlsx', '.json', '.parquet']
    },
    'preprocessing': {
        'numeric_strategies': ['mean', 'median', 'mode', 'constant'],
        'categorical_strategies': ['mode', 'constant'],
        'scaling_methods': ['standard', 'minmax', 'robust']
    },
    'model_training': {
        'default_test_size': 0.2,
        'default_random_state': 42,
        'cross_validation_folds': 5
    },
    'evaluation': {
        'classification_metrics': [
            'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
        ],
        'regression_metrics': [
            'r2', 'mse', 'mae', 'rmse'
        ]
    }
}