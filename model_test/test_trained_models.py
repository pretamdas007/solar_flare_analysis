"""
Comprehensive Model Testing and Analysis with Real XRS Data
Loads trained .h5 models and evaluates them on real solar flare data
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import glob
import os
from pathlib import Path
import warnings
from datetime import datetime, timedelta
import h5py

warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('default')
sns.set_palette("husl")

class ModelTester:
    """
    Comprehensive testing class for all trained solar flare models
    """
    
    def __init__(self, models_dir="models", data_dir="solar_flare_analysis/data/"):
        """
        Initialize the model tester
        
        Parameters
        ----------
        models_dir : str
            Directory containing trained .h5 model files
        data_dir : str
            Directory containing XRS data files
        """
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.models = {}
        self.test_data = {}
        self.results = {}
        
        # Data preprocessing
        self.scaler_X = RobustScaler()
        self.scaler_y = StandardScaler()
        
        print("🚀 Solar Flare Model Tester Initialized")
    
    def load_all_models(self):
        """Load all available trained models"""
        
        # Find all .h5 files in the workspace
        h5_files = []
        
        # Check multiple locations
        search_paths = [
            "*.h5",
            "models/*.h5", 
            "solar_flare_analysis/results/*.h5",
            "../*.h5"
        ]
        
        for pattern in search_paths:
            h5_files.extend(glob.glob(pattern))
        
        print(f"Found {len(h5_files)} .h5 model files:")
        
        for h5_file in h5_files:
            model_name = Path(h5_file).stem
            print(f"  📁 {model_name}: {h5_file}")
            
            try:
                # Try to load the model
                model = keras.models.load_model(h5_file, compile=False)
                self.models[model_name] = {
                    'model': model,
                    'path': h5_file,
                    'loaded': True
                }
                print(f"  ✅ Successfully loaded {model_name}")
                
            except Exception as e:
                print(f"  ❌ Failed to load {model_name}: {str(e)}")
                self.models[model_name] = {
                    'model': None,
                    'path': h5_file,
                    'loaded': False,
                    'error': str(e)
                }
        
        print(f"\\n📊 Successfully loaded {sum(1 for m in self.models.values() if m['loaded'])} models")
        return self.models
    
    def load_real_xrs_data(self, data_pattern="**/*.csv", max_files=10):
        """
        Load real XRS data for testing
        
        Parameters
        ----------
        data_pattern : str
            Glob pattern to find XRS data files
        max_files : int
            Maximum number of files to load
        """
        print("🔍 Loading real XRS data...")
        
        # Find CSV files containing XRS data
        csv_files = glob.glob(data_pattern, recursive=True)
        csv_files = [f for f in csv_files if 'xrs' in f.lower() or 'goes' in f.lower()]
        
        if not csv_files:
            print("⚠️  No XRS data files found. Generating synthetic test data...")
            return self._generate_test_data()
        
        csv_files = csv_files[:max_files]
        print(f"Found {len(csv_files)} XRS data files")
        
        all_data = []
        file_info = []
        
        for file_path in csv_files:
            try:
                print(f"  📄 Loading: {Path(file_path).name}")
                
                # Try different loading methods
                df = None
                
                # Method 1: Standard pandas read
                try:
                    df = pd.read_csv(file_path)
                except:
                    pass
                
                # Method 2: Try with different separators
                if df is None:
                    try:
                        df = pd.read_csv(file_path, sep='\\t')
                    except:
                        try:
                            df = pd.read_csv(file_path, sep=';')
                        except:
                            pass
                
                if df is not None and len(df) > 0:
                    # Look for XRSA and XRSB columns
                    xrsa_col = None
                    xrsb_col = None
                    
                    for col in df.columns:
                        col_lower = col.lower()
                        if 'xrsa' in col_lower or 'a_flux' in col_lower:
                            xrsa_col = col
                        elif 'xrsb' in col_lower or 'b_flux' in col_lower:
                            xrsb_col = col
                    
                    if xrsa_col and xrsb_col:
                        # Extract XRSA and XRSB data
                        xrsa_data = pd.to_numeric(df[xrsa_col], errors='coerce')
                        xrsb_data = pd.to_numeric(df[xrsb_col], errors='coerce')
                        
                        # Remove NaN values
                        valid_mask = ~(np.isnan(xrsa_data) | np.isnan(xrsb_data))
                        xrsa_clean = xrsa_data[valid_mask].values
                        xrsb_clean = xrsb_data[valid_mask].values
                        
                        if len(xrsa_clean) > 100:  # Minimum data length
                            combined_data = np.column_stack([xrsa_clean, xrsb_clean])
                            all_data.append(combined_data)
                            file_info.append({
                                'file': Path(file_path).name,
                                'samples': len(xrsa_clean),
                                'xrsa_col': xrsa_col,
                                'xrsb_col': xrsb_col
                            })
                            print(f"    ✅ Loaded {len(xrsa_clean)} samples")
                        else:
                            print(f"    ⚠️  Insufficient data: {len(xrsa_clean)} samples")
                    else:
                        print(f"    ❌ XRSA/XRSB columns not found")
                else:
                    print(f"    ❌ Failed to load file")
                    
            except Exception as e:
                print(f"    ❌ Error loading {file_path}: {str(e)}")
        
        if all_data:
            self.test_data = {
                'raw_data': all_data,
                'file_info': file_info,
                'sequences': None,
                'labels': None
            }
            print(f"\\n✅ Successfully loaded {len(all_data)} XRS data files")
            return self._prepare_test_sequences()
        else:
            print("⚠️  No valid XRS data loaded. Using synthetic data...")
            return self._generate_test_data()
    
    def _prepare_test_sequences(self, sequence_length=128):
        """Prepare sequences from loaded XRS data"""
        
        if not self.test_data['raw_data']:
            return False
        
        print(f"🔄 Preparing test sequences (length={sequence_length})...")
        
        all_sequences = []
        all_labels = []
        
        for i, data in enumerate(self.test_data['raw_data']):
            print(f"  Processing file {i+1}/{len(self.test_data['raw_data'])}...")
            
            # Create sequences
            sequences = self._create_sequences(data, sequence_length)
            
            # Generate labels (simplified classification based on XRSB flux)
            labels = self._generate_labels_from_data(sequences)
            
            all_sequences.extend(sequences)
            all_labels.extend(labels)
        
        # Convert to numpy arrays
        X_test = np.array(all_sequences)
        y_test = np.array(all_labels)
        
        # Scale the data
        X_test_scaled = self.scaler_X.fit_transform(
            X_test.reshape(-1, X_test.shape[-1])
        ).reshape(X_test.shape)
        
        self.test_data['sequences'] = X_test_scaled
        self.test_data['labels'] = y_test
        
        print(f"✅ Created {len(X_test)} test sequences")
        print(f"   Shape: {X_test.shape}")
        print(f"   Label distribution: {np.bincount(y_test)}")
        
        return True
    
    def _create_sequences(self, data, sequence_length):
        """Create overlapping sequences from time series data"""
        sequences = []
        
        # Create overlapping sequences
        step_size = sequence_length // 4  # 75% overlap
        
        for i in range(0, len(data) - sequence_length + 1, step_size):
            sequence = data[i:i + sequence_length]
            sequences.append(sequence)
        
        return sequences
    
    def _generate_labels_from_data(self, sequences):
        """Generate labels based on XRSB flux levels"""
        labels = []
        
        # GOES flare classification thresholds (W/m²)
        thresholds = {
            0: 0,           # No flare
            1: 1e-8,        # A-class
            2: 1e-7,        # B-class
            3: 1e-6,        # C-class
            4: 1e-5,        # M-class
            5: 1e-4         # X-class
        }
        
        for sequence in sequences:
            # Use maximum XRSB flux in the sequence
            max_flux = np.max(sequence[:, 1])  # XRSB is column 1
            
            # Classify based on thresholds
            label = 0
            for class_num in sorted(thresholds.keys(), reverse=True):
                if max_flux >= thresholds[class_num]:
                    label = class_num
                    break
            
            labels.append(label)
        
        return labels
    
    def _generate_test_data(self, n_samples=1000, sequence_length=128):
        """Generate synthetic test data when real data is not available"""
        print("🔧 Generating synthetic test data...")
        
        X_test = []
        y_test = []
        
        for i in range(n_samples):
            # Generate background
            sequence = np.random.lognormal(-18, 1, (sequence_length, 2))
            
            # Randomly add flares
            if np.random.random() < 0.3:  # 30% chance of flare
                flare_start = np.random.randint(10, sequence_length - 30)
                flare_class = np.random.choice([1, 2, 3, 4, 5], p=[0.5, 0.3, 0.15, 0.04, 0.01])
                
                # Add synthetic flare
                flare_duration = np.random.randint(5, 20)
                flare_end = min(flare_start + flare_duration, sequence_length)
                
                # Flare intensities
                intensities = {1: 1e-8, 2: 1e-7, 3: 1e-6, 4: 1e-5, 5: 1e-4}
                peak_intensity = intensities[flare_class] * np.random.uniform(1, 5)
                
                # Create flare profile
                for j in range(flare_start, flare_end):
                    progress = (j - flare_start) / flare_duration
                    if progress < 0.3:  # Rise phase
                        intensity = peak_intensity * (progress / 0.3)
                    else:  # Decay phase
                        intensity = peak_intensity * np.exp(-(progress - 0.3) / 0.7)
                    
                    sequence[j, 0] += intensity * 0.1  # XRSA
                    sequence[j, 1] += intensity        # XRSB
                
                y_test.append(flare_class)
            else:
                y_test.append(0)  # No flare
            
            X_test.append(sequence)
        
        # Convert to numpy and scale
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # Apply log transform and scaling
        X_test = np.log10(np.maximum(X_test, 1e-20))
        X_test_scaled = self.scaler_X.fit_transform(
            X_test.reshape(-1, X_test.shape[-1])
        ).reshape(X_test.shape)
        
        self.test_data = {
            'sequences': X_test_scaled,
            'labels': y_test,
            'synthetic': True
        }
        
        print(f"✅ Generated {len(X_test)} synthetic test sequences")
        return True
    
    def test_all_models(self):
        """Test all loaded models on the test data"""
        print("🧪 Testing all models...")
        
        if self.test_data.get('sequences') is None:
            print("❌ No test data available. Load data first.")
            return
        
        X_test = self.test_data['sequences']
        y_test = self.test_data['labels']
        
        print(f"Test data shape: {X_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        
        for model_name, model_info in self.models.items():
            if not model_info['loaded']:
                print(f"⏭️  Skipping {model_name} (not loaded)")
                continue
            
            print(f"\\n🔬 Testing {model_name}...")
            
            try:
                model = model_info['model']
                
                # Make predictions
                predictions = model.predict(X_test, verbose=0)
                
                # Handle different output formats
                if isinstance(predictions, list):
                    # Multi-output model
                    if len(predictions) >= 2:
                        # Assume first output is classification, second is regression
                        class_pred = predictions[0]
                        reg_pred = predictions[1] if len(predictions) > 1 else None
                    else:
                        class_pred = predictions[0]
                        reg_pred = None
                else:
                    # Single output
                    if predictions.shape[-1] > 1:
                        # Multi-class classification
                        class_pred = predictions
                        reg_pred = None
                    else:
                        # Regression
                        class_pred = None
                        reg_pred = predictions
                
                # Store results
                self.results[model_name] = {
                    'predictions': predictions,
                    'classification': class_pred,
                    'regression': reg_pred,
                    'test_successful': True
                }
                
                # Calculate metrics
                if class_pred is not None:
                    if class_pred.shape[-1] > 1:
                        # Multi-class
                        y_pred_class = np.argmax(class_pred, axis=1)
                    else:
                        # Binary
                        y_pred_class = (class_pred > 0.5).astype(int).flatten()
                    
                    # Calculate accuracy
                    accuracy = np.mean(y_pred_class == y_test)
                    self.results[model_name]['accuracy'] = accuracy
                    self.results[model_name]['y_pred_class'] = y_pred_class
                    
                    print(f"  📊 Classification Accuracy: {accuracy:.3f}")
                
                if reg_pred is not None:
                    # Calculate regression metrics
                    mse = mean_squared_error(y_test, reg_pred.flatten())
                    r2 = r2_score(y_test, reg_pred.flatten())
                    self.results[model_name]['mse'] = mse
                    self.results[model_name]['r2'] = r2
                    
                    print(f"  📈 Regression MSE: {mse:.6f}")
                    print(f"  📈 Regression R²: {r2:.3f}")
                
                print(f"  ✅ {model_name} tested successfully")
                
            except Exception as e:
                print(f"  ❌ Error testing {model_name}: {str(e)}")
                self.results[model_name] = {
                    'test_successful': False,
                    'error': str(e)
                }
        
        print(f"\\n🎯 Testing complete! Results for {len(self.results)} models available.")
    
    def create_comprehensive_analysis(self):
        """Create comprehensive analysis plots and reports"""
        print("📊 Creating comprehensive analysis...")
        
        # Create output directory
        output_dir = Path("model_test_results")
        output_dir.mkdir(exist_ok=True)
        
        # 1. Model Performance Comparison
        self._plot_model_comparison(output_dir)
        
        # 2. Individual Model Analysis
        self._plot_individual_analyses(output_dir)
        
        # 3. Prediction Visualizations
        self._plot_prediction_examples(output_dir)
        
        # 4. Generate Report
        self._generate_analysis_report(output_dir)
        
        print(f"✅ Analysis complete! Results saved in: {output_dir}")
    
    def _plot_model_comparison(self, output_dir):
        """Plot comparison of all model performances"""
        
        successful_models = {name: results for name, results in self.results.items() 
                           if results.get('test_successful', False)}
        
        if not successful_models:
            print("⚠️  No successful model results to compare")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract metrics
        model_names = list(successful_models.keys())
        accuracies = [results.get('accuracy', 0) for results in successful_models.values()]
        mses = [results.get('mse', 0) for results in successful_models.values()]
        r2s = [results.get('r2', 0) for results in successful_models.values()]
        
        # 1. Accuracy comparison
        axes[0, 0].bar(range(len(model_names)), accuracies, alpha=0.7)
        axes[0, 0].set_title('Model Classification Accuracy', fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks(range(len(model_names)))
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. MSE comparison (if available)
        if any(mse > 0 for mse in mses):
            axes[0, 1].bar(range(len(model_names)), mses, alpha=0.7, color='orange')
            axes[0, 1].set_title('Model Regression MSE', fontweight='bold')
            axes[0, 1].set_ylabel('Mean Squared Error')
            axes[0, 1].set_xticks(range(len(model_names)))
            axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. R² comparison (if available)
        if any(r2 != 0 for r2 in r2s):
            axes[1, 0].bar(range(len(model_names)), r2s, alpha=0.7, color='green')
            axes[1, 0].set_title('Model Regression R²', fontweight='bold')
            axes[1, 0].set_ylabel('R² Score')
            axes[1, 0].set_xticks(range(len(model_names)))
            axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Model summary table
        axes[1, 1].axis('off')
        table_data = []
        for name, results in successful_models.items():
            row = [
                name[:15] + '...' if len(name) > 15 else name,
                f"{results.get('accuracy', 0):.3f}",
                f"{results.get('mse', 0):.2e}" if results.get('mse', 0) > 0 else 'N/A',
                f"{results.get('r2', 0):.3f}" if results.get('r2', 0) != 0 else 'N/A'
            ]
            table_data.append(row)
        
        table = axes[1, 1].table(cellText=table_data,
                               colLabels=['Model', 'Accuracy', 'MSE', 'R²'],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('Model Performance Summary', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✅ Model comparison plot saved")
    
    def _plot_individual_analyses(self, output_dir):
        """Create individual analysis plots for each model"""
        
        for model_name, results in self.results.items():
            if not results.get('test_successful', False):
                continue
            
            if 'y_pred_class' not in results:
                continue
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'{model_name} - Detailed Analysis', fontsize=14, fontweight='bold')
            
            y_true = self.test_data['labels']
            y_pred = results['y_pred_class']
            
            # 1. Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues')
            axes[0, 0].set_title('Confusion Matrix')
            axes[0, 0].set_xlabel('Predicted')
            axes[0, 0].set_ylabel('Actual')
            
            # 2. Class distribution comparison
            unique_true = np.bincount(y_true, minlength=6)
            unique_pred = np.bincount(y_pred, minlength=6)
            
            x_pos = np.arange(6)
            width = 0.35
            
            axes[0, 1].bar(x_pos - width/2, unique_true, width, label='True', alpha=0.7)
            axes[0, 1].bar(x_pos + width/2, unique_pred, width, label='Predicted', alpha=0.7)
            axes[0, 1].set_title('Class Distribution Comparison')
            axes[0, 1].set_xlabel('Flare Class')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels(['None', 'A', 'B', 'C', 'M', 'X'])
            axes[0, 1].legend()
            
            # 3. Prediction confidence (if available)
            if 'classification' in results and results['classification'] is not None:
                class_probs = results['classification']
                if class_probs.shape[-1] > 1:
                    max_probs = np.max(class_probs, axis=1)
                    axes[1, 0].hist(max_probs, bins=20, alpha=0.7, edgecolor='black')
                    axes[1, 0].set_title('Prediction Confidence Distribution')
                    axes[1, 0].set_xlabel('Max Probability')
                    axes[1, 0].set_ylabel('Frequency')
            
            # 4. Performance by class
            report = classification_report(y_true, y_pred, output_dict=True)
            classes = ['0', '1', '2', '3', '4', '5']
            f1_scores = [report.get(cls, {}).get('f1-score', 0) for cls in classes]
            
            axes[1, 1].bar(range(6), f1_scores, alpha=0.7)
            axes[1, 1].set_title('F1-Score by Class')
            axes[1, 1].set_xlabel('Flare Class')
            axes[1, 1].set_ylabel('F1-Score')
            axes[1, 1].set_xticks(range(6))
            axes[1, 1].set_xticklabels(['None', 'A', 'B', 'C', 'M', 'X'])
            
            plt.tight_layout()
            plt.savefig(output_dir / f"{model_name}_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print("  ✅ Individual model analyses saved")
    
    def _plot_prediction_examples(self, output_dir):
        """Plot example predictions from the best performing model"""
        
        # Find best model
        best_model = None
        best_accuracy = 0
        
        for model_name, results in self.results.items():
            if results.get('test_successful', False) and results.get('accuracy', 0) > best_accuracy:
                best_accuracy = results.get('accuracy', 0)
                best_model = model_name
        
        if not best_model:
            print("  ⚠️  No successful models found for prediction examples")
            return
        
        print(f"  📈 Creating prediction examples for best model: {best_model}")
        
        X_test = self.test_data['sequences']
        y_true = self.test_data['labels']
        y_pred = self.results[best_model]['y_pred_class']
        
        # Select interesting examples
        correct_pred = y_true == y_pred
        incorrect_pred = y_true != y_pred
        
        # Find examples of each type
        correct_indices = np.where(correct_pred)[0][:4]
        incorrect_indices = np.where(incorrect_pred)[0][:4]
        
        fig, axes = plt.subplots(4, 2, figsize=(15, 12))
        fig.suptitle(f'Prediction Examples - {best_model}', fontsize=14, fontweight='bold')
        
        class_names = ['None', 'A', 'B', 'C', 'M', 'X']
        
        # Plot correct predictions
        for i, idx in enumerate(correct_indices):
            if i >= 4:
                break
            
            sequence = X_test[idx]
            axes[i, 0].plot(sequence[:, 0], label='XRSA', alpha=0.7)
            axes[i, 0].plot(sequence[:, 1], label='XRSB', alpha=0.7)
            axes[i, 0].set_title(f'✅ Correct: True={class_names[y_true[idx]]}, Pred={class_names[y_pred[idx]]}')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
        
        # Plot incorrect predictions
        for i, idx in enumerate(incorrect_indices):
            if i >= 4:
                break
            
            sequence = X_test[idx]
            axes[i, 1].plot(sequence[:, 0], label='XRSA', alpha=0.7)
            axes[i, 1].plot(sequence[:, 1], label='XRSB', alpha=0.7)
            axes[i, 1].set_title(f'❌ Incorrect: True={class_names[y_true[idx]]}, Pred={class_names[y_pred[idx]]}')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"prediction_examples_{best_model}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✅ Prediction examples saved")
    
    def _generate_analysis_report(self, output_dir):
        """Generate a comprehensive text report"""
        
        report_content = f"""
# Solar Flare Model Testing Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Test Data Summary
- Total sequences: {len(self.test_data['sequences']) if self.test_data.get('sequences') is not None else 'N/A'}
- Sequence length: {self.test_data['sequences'].shape[1] if self.test_data.get('sequences') is not None else 'N/A'}
- Features: {self.test_data['sequences'].shape[2] if self.test_data.get('sequences') is not None else 'N/A'}
- Data source: {'Synthetic' if self.test_data.get('synthetic', False) else 'Real XRS data'}

## Label Distribution
"""
        
        if self.test_data.get('labels') is not None:
            label_counts = np.bincount(self.test_data['labels'], minlength=6)
            class_names = ['None', 'A-class', 'B-class', 'C-class', 'M-class', 'X-class']
            
            for i, (name, count) in enumerate(zip(class_names, label_counts)):
                percentage = (count / len(self.test_data['labels'])) * 100
                report_content += f"- {name}: {count} ({percentage:.1f}%)\\n"
        
        report_content += f"""
## Model Performance Summary

| Model | Status | Accuracy | MSE | R² | Notes |
|-------|--------|----------|-----|----|----- |
"""
        
        for model_name, results in self.results.items():
            if results.get('test_successful', False):
                accuracy = results.get('accuracy', 0)
                mse = results.get('mse', 0)
                r2 = results.get('r2', 0)
                report_content += f"| {model_name} | ✅ Success | {accuracy:.3f} | {mse:.2e} | {r2:.3f} | - |\\n"
            else:
                error = results.get('error', 'Unknown error')
                report_content += f"| {model_name} | ❌ Failed | - | - | - | {error[:50]}... |\\n"
        
        # Find best performing model
        best_models = {name: results for name, results in self.results.items() 
                      if results.get('test_successful', False)}
        
        if best_models:
            best_model = max(best_models.items(), key=lambda x: x[1].get('accuracy', 0))
            report_content += f"""
## Best Performing Model
**{best_model[0]}**
- Accuracy: {best_model[1].get('accuracy', 0):.3f}
- MSE: {best_model[1].get('mse', 0):.2e}
- R²: {best_model[1].get('r2', 0):.3f}

## Recommendations
1. The {best_model[0]} shows the best overall performance for solar flare classification
2. Consider ensemble methods combining multiple models for improved robustness
3. Evaluate models on additional real-world datasets for validation
4. Monitor model performance on different flare types and intensities

## Files Generated
- model_comparison.png: Overall model performance comparison
- {best_model[0]}_analysis.png: Detailed analysis of best model
- prediction_examples_{best_model[0]}.png: Example predictions
- analysis_report.txt: This report
"""
        
        # Save report
        with open(output_dir / "analysis_report.txt", 'w') as f:
            f.write(report_content)
        
        print("  ✅ Analysis report saved")

def main():
    """Main function to run the comprehensive model testing"""
    
    print("🌟 Solar Flare Model Testing and Analysis")
    print("=" * 50)
    
    # Initialize tester
    tester = ModelTester()
    
    # Load all available models
    tester.load_all_models()
    
    # Load test data (try real XRS data first, fallback to synthetic)
    tester.load_real_xrs_data()
    
    # Test all models
    tester.test_all_models()
    
    # Create comprehensive analysis
    tester.create_comprehensive_analysis()
    
    print("\\n🎉 Testing and analysis complete!")
    print("Check the 'model_test_results' directory for all outputs.")

if __name__ == "__main__":
    main()
