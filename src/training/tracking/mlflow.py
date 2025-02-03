import mlflow
from typing import Any
from datetime import datetime

class MLFlowTracker:
    """MLflow experiment tracking integration"""
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.run_name = f"{config['model']}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
    def __enter__(self):
        mlflow.start_run(run_name=self.run_name)
        mlflow.log_params(self.config)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow.end_run()
    
    def log_metrics(self, metrics: dict[str, float], step: int):
        mlflow.log_metrics(metrics, step=step)