"""
Monte Carlo Solar Flare Model - Professional Algorithm Flowchart
Professional black and white version for publication-ready output
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Set professional style
plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14

def create_monte_carlo_flowchart():
    """Create professional black and white Monte Carlo algorithm flowchart"""
    
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 18)
    ax.axis('off')
    
    # Professional black and white color scheme
    colors = {
        'input': 'white',
        'preprocessing': '#F5F5F5', 
        'model': '#E8E8E8',
        'monte_carlo': '#D3D3D3',
        'bayesian': '#C0C0C0',
        'output': '#F0F0F0',
        'uncertainty': '#DCDCDC'
    }
    
    # Professional border styles
    border_styles = {
        'thick': 2.5,
        'medium': 2.0,
        'thin': 1.5
    }      # Title
    ax.text(5, 17.5, 'Monte Carlo Solar Flare Model Algorithm', 
            fontsize=24, fontweight='bold', ha='center', color='black')
    ax.text(5, 17, 'Uncertainty Quantification for Solar Flare Prediction', 
            fontsize=18, ha='center', style='italic', color='black')
      # Step 1: Data Input
    step1 = FancyBboxPatch((3.5, 15.5), 3, 1, boxstyle="round,pad=0.1", 
                          facecolor=colors['input'], edgecolor='black', linewidth=border_styles['thick'])
    ax.add_patch(step1)
    ax.text(5, 16, 'XRS Time Series Data\n(XRSA, XRSB channels)', 
            ha='center', va='center', fontsize=14, fontweight='bold', color='black')
    
    # Step 2: Preprocessing
    step2 = FancyBboxPatch((3.5, 14), 3, 1, boxstyle="round,pad=0.1", 
                          facecolor=colors['preprocessing'], edgecolor='black', linewidth=border_styles['medium'])
    ax.add_patch(step2)
    ax.text(5, 14.5, 'Data Preprocessing\nLog transform & Scaling', 
            ha='center', va='center', fontsize=14, fontweight='bold', color='black')
    
    # Step 3: Neural Network
    step3 = FancyBboxPatch((3.5, 12.5), 3, 1.5, boxstyle="round,pad=0.1", 
                          facecolor=colors['model'], edgecolor='black', linewidth=border_styles['thick'])
    ax.add_patch(step3)
    ax.text(5, 13.25, 'Neural Network\nLSTM + Dense Layers\nMonte Carlo Dropout', 
            ha='center', va='center', fontsize=14, fontweight='bold', color='black')
    
    # Step 4: Bayesian Layers
    step4 = FancyBboxPatch((3.5, 10.5), 3, 1.5, boxstyle="round,pad=0.1", 
                          facecolor=colors['bayesian'], edgecolor='black', linewidth=border_styles['thick'])
    ax.add_patch(step4)
    ax.text(5, 11.25, 'Bayesian Layers\nWeight Distributions\nVariational Inference', 
            ha='center', va='center', fontsize=14, fontweight='bold', color='black')
    
    # Step 5: Monte Carlo Sampling
    step5 = FancyBboxPatch((3.5, 8.5), 3, 1.5, boxstyle="round,pad=0.1", 
                          facecolor=colors['monte_carlo'], edgecolor='black', linewidth=border_styles['thick'])
    ax.add_patch(step5)
    ax.text(5, 9.25, 'Monte Carlo Sampling\nN Forward Passes\nDropout Active', 
            ha='center', va='center', fontsize=14, fontweight='bold', color='black')      # Step 6: Multi-task Outputs
    step6a = FancyBboxPatch((1, 6.5), 2.5, 1, boxstyle="round,pad=0.1", 
                           facecolor=colors['output'], edgecolor='black', linewidth=border_styles['medium'])
    ax.add_patch(step6a)
    ax.text(2.25, 7, 'Detection\nBinary Output', 
            ha='center', va='center', fontsize=13, fontweight='bold', color='black')
    
    step6b = FancyBboxPatch((3.75, 6.5), 2.5, 1, boxstyle="round,pad=0.1", 
                           facecolor=colors['output'], edgecolor='black', linewidth=border_styles['medium'])
    ax.add_patch(step6b)
    ax.text(5, 7, 'Classification\nFlare Classes', 
            ha='center', va='center', fontsize=13, fontweight='bold', color='black')
    
    step6c = FancyBboxPatch((6.5, 6.5), 2.5, 1, boxstyle="round,pad=0.1", 
                           facecolor=colors['output'], edgecolor='black', linewidth=border_styles['medium'])
    ax.add_patch(step6c)
    ax.text(7.75, 7, 'Regression\nIntensity', 
            ha='center', va='center', fontsize=13, fontweight='bold', color='black')
    
    # Step 7: Uncertainty Quantification
    step7 = FancyBboxPatch((2, 4.5), 6, 1.5, boxstyle="round,pad=0.1", 
                          facecolor=colors['uncertainty'], edgecolor='black', linewidth=border_styles['thick'])
    ax.add_patch(step7)
    ax.text(5, 5.25, 'Uncertainty Quantification\nEpistemic + Aleatoric\nConfidence Intervals', 
            ha='center', va='center', fontsize=14, fontweight='bold', color='black')
    
    # Step 8: Final Predictions
    step8 = FancyBboxPatch((3.5, 2.5), 3, 1, boxstyle="round,pad=0.1", 
                          facecolor=colors['output'], edgecolor='black', linewidth=border_styles['thick'])
    ax.add_patch(step8)
    ax.text(5, 3, 'Predictions with\nUncertainty Bounds', 
            ha='center', va='center', fontsize=14, fontweight='bold', color='black')
      # Add professional arrows
    arrows = [
        ((5, 15.5), (5, 15)),      # Input to preprocessing
        ((5, 14), (5, 14)),        # Preprocessing to NN
        ((5, 12.5), (5, 12)),      # NN to Bayesian
        ((5, 10.5), (5, 10)),      # Bayesian to MC sampling
        ((5, 8.5), (2.25, 7.5)),   # MC to detection
        ((5, 8.5), (5, 7.5)),      # MC to classification
        ((5, 8.5), (7.75, 7.5)),   # MC to regression
        ((2.25, 6.5), (3, 6)),     # Detection to uncertainty
        ((5, 6.5), (5, 6)),        # Classification to uncertainty
        ((7.75, 6.5), (7, 6)),     # Regression to uncertainty
        ((5, 4.5), (5, 3.5))       # Uncertainty to final
    ]
    
    for (start, end) in arrows:
        arrow = ConnectionPatch(start, end, "data", "data", 
                              arrowstyle="-|>", shrinkA=5, shrinkB=5,
                              mutation_scale=25, fc="black", ec="black", linewidth=2)
        ax.add_patch(arrow)
      # Professional mathematical formulation box
    math_box = FancyBboxPatch((0.5, 0.5), 9, 1.5, boxstyle="round,pad=0.15", 
                             facecolor='white', edgecolor='black', linewidth=border_styles['medium'])
    ax.add_patch(math_box)
    
    math_text = """Mathematical Foundation:
Monte Carlo Estimation: E[f(x)] ≈ (1/N) Σᵢ₌₁ᴺ f(x, θᵢ) where θᵢ ~ q(θ)
Bayesian Inference: P(θ|D) = P(D|θ)P(θ) / P(D)
Uncertainty Decomposition: σ²total = σ²epistemic + σ²aleatoric"""
    
    ax.text(5, 1.25, math_text, ha='center', va='center', fontsize=12, 
            fontfamily='monospace', color='black',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    return fig

def main():
    """Generate professional black and white Monte Carlo algorithm flowchart"""
    
    print("Generating Professional Monte Carlo Solar Flare Model Algorithm Flowchart...")
    
    fig = create_monte_carlo_flowchart()
    fig.savefig("monte_carlo_algorithm_flowchart_professional.png", 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='black')
    plt.close(fig)
    
    print("SUCCESS: Professional Monte Carlo algorithm flowchart saved: monte_carlo_algorithm_flowchart_professional.png")

if __name__ == "__main__":
    main()
