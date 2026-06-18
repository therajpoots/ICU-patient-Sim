"""
ML CLI Runner
Launches the ICU Biosignal Anomaly Detection Pipeline.
Usage:
    python run_ml.py --train --output-dir "C:/path/to/artifacts"
"""

import os
import argparse
from ml.train import run_ml_pipeline

def main():
    parser = argparse.ArgumentParser(description="ICU Biosignal Anomaly Detection CLI Runner")
    parser.add_argument("--train", action="store_true", help="Train and evaluate all ML models")
    parser.add_argument("--duration", type=int, default=180, help="Simulation duration (s) per patient state (default 180)")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save the xAI surrogate tree plot (default current dir)")
    
    args = parser.parse_args()
    
    if args.train:
        # Create output directory if it doesn't exist
        if args.output_dir and not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
            
        print(f"Starting ML anomaly detection training pipeline...")
        print(f"Simulation duration per state: {args.duration}s")
        print(f"Saving artifacts to: {args.output_dir}")
        
        run_ml_pipeline(duration_per_state=args.duration, output_dir=args.output_dir)
        print("Training pipeline run complete.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
