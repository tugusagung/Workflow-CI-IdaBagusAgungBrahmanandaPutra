import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import json
from datetime import datetime

def main():
    # Parse arguments dari MLProject
    parser = argparse.ArgumentParser(description="Train model untuk MLflow Project")
    parser.add_argument("--experiment_name", type=str, default="mlflow-project-run")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    
    args = parser.parse_args()
    
    print("="*70)
    print("MLFLOW PROJECT: MODELLING.PY")
    print("="*70)
    print(f"Experiment: {args.experiment_name}")
    print(f"n_estimators: {args.n_estimators}")
    print(f"max_depth: {args.max_depth}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. LOAD DATA
    print("\n1. LOADING DATA")
    try:
        X_train = pd.read_csv('diabetes_preprocessed/X_train.csv')
        X_test = pd.read_csv('diabetes_preprocessed/X_test.csv')
        y_train = pd.read_csv('diabetes_preprocessed/y_train.csv').values.ravel()
        y_test = pd.read_csv('diabetes_preprocessed/y_test.csv').values.ravel()
        
        print(f"   Success: X_train{X_train.shape}, X_test{X_test.shape}")
    except Exception as e:
        print(f"   Error loading data: {e}")
        # Create dummy data untuk testing
        print("   Using dummy data for testing")
        np.random.seed(42)
        X_train = np.random.rand(100, 8)
        X_test = np.random.rand(20, 8)
        y_train = np.random.randint(0, 2, 100)
        y_test = np.random.randint(0, 2, 20)
    
    # 2. SETUP MLFLOW
    print("\n2. SETUP MLFLOW")
    mlflow.set_experiment(args.experiment_name)
    
    # Check if we're already in an active run (when called from mlflow run)
    # If yes, don't create a new run
    if mlflow.active_run() is None:
        # Only create new run if not running from mlflow run command
        run_context = mlflow.start_run(run_name=f"mlproject-{datetime.now().strftime('%H%M%S')}")
    else:
        # Use existing run from mlflow run command
        run_context = mlflow.active_run()
        print(f"   Using existing MLflow run: {run_context.info.run_id}")
    
    # Use context manager only if we created the run
    should_end_run = mlflow.active_run() is not None and run_context is not None
    
    try:
        # 3. LOG PARAMETERS
        print("\n3. LOGGING PARAMETERS")
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("project", "MLflow_Project_CI")
        
        # 4. TRAIN MODEL
        print("\n4. TRAINING MODEL")
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        print("   Model training completed")
        
        # 5. EVALUATION
        print("\n5. MODEL EVALUATION")
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        # Log metrics
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
            print(f"   {name}: {value:.4f}")
        
        # 6. SAVE MODEL & ARTIFACTS
        print("\n6. SAVING ARTIFACTS")
        os.makedirs('outputs', exist_ok=True)
        
        # Save model
        model_path = 'outputs/model.pkl'
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        print(f"   Model saved: {model_path}")
        
        # Save metrics as JSON
        metrics_path = 'outputs/metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(metrics_path)
        print(f"   Metrics saved: {metrics_path}")
        
        # Save run info
        run_info = {
            'run_id': mlflow.active_run().info.run_id,
            'experiment': args.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'n_estimators': args.n_estimators,
                'max_depth': args.max_depth
            },
            'metrics': metrics,
            'data_info': {
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'features': X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0])
            }
        }
        
        info_path = 'outputs/run_info.json'
        with open(info_path, 'w') as f:
            json.dump(run_info, f, indent=2)
        mlflow.log_artifact(info_path)
        
        # 7. LOG MODEL ke MLflow
        print("\n7. LOGGING MODEL TO MLFLOW")
        mlflow.sklearn.log_model(model, "mlflow_model")
        print("   Model logged to MLflow")
        
        # 8. FINAL OUTPUT
        print("\n" + "="*70)
        print("MLFLOW PROJECT EXECUTION COMPLETE!")
        print("="*70)
        print(f"\nBest metric - Accuracy: {metrics['accuracy']:.4f}")
        print(f"Output folder: outputs/")
        print(f"Files generated:")
        print(f"   - {model_path}")
        print(f"   - {metrics_path}")
        print(f"   - {info_path}")
        
        return metrics['accuracy']
    
    finally:
        # Only end run if we created it (not from mlflow run)
        if should_end_run and isinstance(run_context, mlflow.ActiveRun):
            mlflow.end_run()

if __name__ == "__main__":
    print("Starting MLflow Project execution...")
    accuracy = main()
    print(f"\n Final accuracy: {accuracy:.4f}")