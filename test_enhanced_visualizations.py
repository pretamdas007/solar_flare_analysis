#!/usr/bin/env python3
"""
Test script for enhanced seaborn visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def test_enhanced_seaborn_styling():
    """Test the enhanced seaborn professional styling"""
    print("Testing Enhanced Seaborn Professional Visualizations...")
    
    # Set professional seaborn styling
    plt.style.use('seaborn-v0_8')
    sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
    sns.set_context("paper", rc={"figure.dpi": 300})
    
    # Professional color palettes
    primary_palette = sns.color_palette("viridis", 8)
    accent_palette = sns.color_palette("rocket", 6)
    diverging_palette = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=False)
    
    # Create test figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='white')
    
    # 1. Professional Time Series Plot
    ax1 = axes[0, 0]
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x) + np.random.normal(0, 0.1, 100)
    y2 = np.cos(x) + np.random.normal(0, 0.1, 100)
    
    ts_data = pd.DataFrame({
        'Time': np.tile(x, 2),
        'Value': np.concatenate([y1, y2]),
        'Series': ['XRS-A'] * 100 + ['XRS-B'] * 100
    })
    
    sns.lineplot(data=ts_data, x='Time', y='Value', hue='Series', 
                ax=ax1, linewidth=2.5, alpha=0.8, palette=primary_palette[:2])
    ax1.set_title('Professional Time Series', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')
    
    # 2. Enhanced Distribution Plot
    ax2 = axes[0, 1]
    dist_data = pd.DataFrame({
        'Values': np.concatenate([np.random.normal(0, 1, 100), np.random.normal(2, 1.5, 100)]),
        'Category': ['A'] * 100 + ['B'] * 100
    })
    
    sns.violinplot(data=dist_data, x='Category', y='Values', ax=ax2, 
                  palette=accent_palette[:2], alpha=0.8, inner='quart')
    ax2.set_title('Enhanced Distribution Analysis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Advanced Correlation Matrix
    ax3 = axes[0, 2]
    corr_data = np.random.randn(50, 5)
    corr_df = pd.DataFrame(corr_data, columns=['Feature A', 'Feature B', 'Feature C', 'Feature D', 'Feature E'])
    correlation_matrix = corr_df.corr()
    
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
               center=0, square=True, ax=ax3, cbar_kws={'shrink': 0.8})
    ax3.set_title('Advanced Correlation Matrix', fontsize=14, fontweight='bold')
    
    # 4. Professional Box Plot
    ax4 = axes[1, 0]
    box_data = pd.DataFrame({
        'Intensity': np.concatenate([np.random.exponential(2, 100), np.random.exponential(3, 100)]),
        'Type': ['Background'] * 100 + ['Flare'] * 100
    })
    
    sns.boxplot(data=box_data, x='Type', y='Intensity', ax=ax4, palette=primary_palette[:2])
    sns.stripplot(data=box_data, x='Type', y='Intensity', ax=ax4, 
                 size=3, alpha=0.6, color='black', jitter=True)
    ax4.set_title('Professional Box Plot Analysis', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Performance Heatmap
    ax5 = axes[1, 1]
    performance_data = np.random.uniform(0.5, 0.95, (5, 4))
    perf_df = pd.DataFrame(performance_data, 
                          columns=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                          index=['Model A', 'Model B', 'Model C', 'Model D', 'Model E'])
    
    sns.heatmap(perf_df, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax5,
               cbar_kws={'label': 'Performance Score'})
    ax5.set_title('Performance Matrix', fontsize=14, fontweight='bold')
    
    # 6. Scatter Plot with Regression
    ax6 = axes[1, 2]
    scatter_data = pd.DataFrame({
        'X': np.random.normal(0, 1, 100),
        'Y': np.random.normal(0, 1, 100) + np.random.normal(0, 0.5, 100),
        'Category': np.random.choice(['Type 1', 'Type 2'], 100)
    })
    
    sns.scatterplot(data=scatter_data, x='X', y='Y', hue='Category', 
                   ax=ax6, alpha=0.7, s=50, palette=accent_palette[:2])
    sns.regplot(data=scatter_data, x='X', y='Y', ax=ax6, scatter=False, color='red')
    ax6.set_title('Scatter Plot with Regression', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('🎨 Professional Seaborn Visualization Test Suite', 
                fontsize=18, fontweight='bold', y=0.95,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightsteelblue', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Save the test visualization
    output_dir = Path("enhanced_output")
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'test_seaborn_professional.png', 
               dpi=300, bbox_inches='tight', facecolor='white', 
               edgecolor='none', format='png')
    plt.show()
    
    print("✅ Enhanced seaborn visualization test completed!")
    print(f"📁 Test output saved to: {output_dir / 'test_seaborn_professional.png'}")
    
    return True

if __name__ == "__main__":
    test_enhanced_seaborn_styling()
