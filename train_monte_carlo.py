
#!/usr/bin/env python3
"""
Dedicated Training Script for the Enhanced Monte Carlo Solar Flare Model
"""

import sys
import logging
import argparse
from solar_flare_analysis.src.ml_models.monte_carlo_enhanced_model import MonteCarloSolarFlareModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monte_carlo_training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to train the Monte Carlo model."""
    parser = argparse.ArgumentParser(description='Train the Monte Carlo Solar Flare Model.')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate for Monte Carlo uncertainty.')
    parser.add_argument('--mc_samples', type=int, default=100, help='Number of Monte Carlo samples for uncertainty estimation.')
    parser.add_argument('--sequence_length', type=int, default=128, help='Length of input time series sequences.')
    parser.add_argument('--data_dir', type=str, default='solar_flare_analysis/data/XRS', help='Directory containing XRS CSV data files.')
    parser.add_argument('--max_files', type=int, default=5, help='Maximum number of XRS files to load for training.')
    parser.add_argument('--output_model_path', type=str, default='monte_carlo_model.h5', help='Path to save the trained model.')
    parser.add_argument('--output_plot_path', type=str, default='monte_carlo_training_analysis.png', help='Path to save the training analysis plot.')

    args = parser.parse_args()

    logger.info("=================================================")
    logger.info("Starting Training for Monte Carlo Solar Flare Model")
    logger.info("=================================================")
    logger.info(f"Training Parameters: {vars(args)}")

    # 1. Initialize the model with parameters from the command line
    mc_model = MonteCarloSolarFlareModel(
        sequence_length=args.sequence_length,
        n_features=2,  # XRSA and XRSB
        n_classes=6,   # No Flare, A, B, C, M, X
        mc_samples=args.mc_samples,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate
    )

    # 2. Build the Keras model structure
    mc_model.build_monte_carlo_model()
    mc_model.model.summary()

    # 3. Train the model
    # The train_model method handles data loading, preprocessing, splitting, and training
    try:
        history = mc_model.train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_callbacks=True
        )
    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")
        logger.error("Attempting to proceed with synthetic data for demonstration purposes.")
        # Fallback to synthetic data if real data loading/training fails
        mc_model._generate_synthetic_training_data()
        history = mc_model.train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_callbacks=False # Disable callbacks for synthetic data run
        )


    logger.info("Model training completed.")

    # 4. Evaluate the model on validation data
    logger.info("Evaluating model performance...")
    evaluation_results = mc_model.evaluate_model()
    logger.info(f"Evaluation Results: {evaluation_results}")

    # 5. Generate and save diagnostic plots
    logger.info("Generating training history plot...")
    mc_model.plot_training_history(history.history, save_path=args.output_plot_path)

    # 6. Save the final trained model
    logger.info(f"Saving trained model to {args.output_model_path}...")
    mc_model.save_model(args.output_model_path)

    logger.info("=================================================")
    logger.info("Monte Carlo Model Training Script Finished")
    logger.info("=================================================")

if __name__ == "__main__":
    main()
