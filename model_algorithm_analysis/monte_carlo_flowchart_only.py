"""
Monte Carlo Solar Flare Model - Professional Algorithm Flowchart
Simplified version generating only the main algorithm flowchart
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import seaborn as sns
import numpy as np

# Set style
plt.style.use('default')
sns.set_palette("husl")

def create_monte_carlo_flowchart():
    """Create professional Monte Carlo algorithm flowchart"""
    
    fig, ax = plt.subplots(figsize=(14, 18))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 18)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#E8F4FD',
        'preprocessing': '#B8E6B8', 
        'model': '#FFD700',
        'monte_carlo': '#FF6B6B',
        'bayesian': '#9370DB',
        'output': '#FFA07A',
        'uncertainty': '#20B2AA'
    }
    
    # Title
    ax.text(5, 17.5, 'Monte Carlo Solar Flare Model Algorithm', 
            fontsize=18, fontweight='bold', ha='center')
    ax.text(5, 17, 'Uncertainty Quantification for Solar Flare Prediction', 
            fontsize=14, ha='center', style='italic')
    
    # Step 1: Data Input
    step1 = FancyBboxPatch((3.5, 15.5), 3, 1, boxstyle="round,pad=0.1", 
                          facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(step1)
    ax.text(5, 16, 'XRS Time Series Data\n(XRSA, XRSB channels)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 2: Preprocessing
    step2 = FancyBboxPatch((3.5, 14), 3, 1, boxstyle="round,pad=0.1", 
                          facecolor=colors['preprocessing'], edgecolor='black', linewidth=2)
    ax.add_patch(step2)
    ax.text(5, 14.5, 'Data Preprocessing\nLog transform & Scaling', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 3: Neural Network
    step3 = FancyBboxPatch((3.5, 12.5), 3, 1.5, boxstyle="round,pad=0.1", 
                          facecolor=colors['model'], edgecolor='black', linewidth=2)
    ax.add_patch(step3)
    ax.text(5, 13.25, 'Neural Network\nLSTM + Dense Layers\nMonte Carlo Dropout', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 4: Bayesian Layers
    step4 = FancyBboxPatch((3.5, 10.5), 3, 1.5, boxstyle="round,pad=0.1", 
                          facecolor=colors['bayesian'], edgecolor='black', linewidth=2)
    ax.add_patch(step4)
    ax.text(5, 11.25, 'Bayesian Layers\nWeight Distributions\nVariational Inference', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 5: Monte Carlo Sampling
    step5 = FancyBboxPatch((3.5, 8.5), 3, 1.5, boxstyle="round,pad=0.1", 
                          facecolor=colors['monte_carlo'], edgecolor='black', linewidth=2)
    ax.add_patch(step5)
    ax.text(5, 9.25, 'Monte Carlo Sampling\nN Forward Passes\nDropout Active', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 6: Multi-task Outputs
    step6a = FancyBboxPatch((1, 6.5), 2.5, 1, boxstyle="round,pad=0.1", 
                           facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(step6a)
    ax.text(2.25, 7, 'Detection\nBinary Output', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    step6b = FancyBboxPatch((3.75, 6.5), 2.5, 1, boxstyle="round,pad=0.1", 
                           facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(step6b)
    ax.text(5, 7, 'Classification\nFlare Classes', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    step6c = FancyBboxPatch((6.5, 6.5), 2.5, 1, boxstyle="round,pad=0.1", 
                           facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(step6c)
    ax.text(7.75, 7, 'Regression\nIntensity', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Step 7: Uncertainty Quantification
    step7 = FancyBboxPatch((2, 4.5), 6, 1.5, boxstyle="round,pad=0.1", 
                          facecolor=colors['uncertainty'], edgecolor='black', linewidth=2)
    ax.add_patch(step7)
    ax.text(5, 5.25, 'Uncertainty Quantification\nEpistemic + Aleatoric\nConfidence Intervals', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Step 8: Final Predictions
    step8 = FancyBboxPatch((3.5, 2.5), 3, 1, boxstyle="round,pad=0.1", 
                          facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(step8)
    ax.text(5, 3, 'Predictions with\nUncertainty Bounds', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Add arrows
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
                              mutation_scale=20, fc="black")
        ax.add_patch(arrow)
    
    # Mathematical formulation
    math_box = FancyBboxPatch((0.5, 0.5), 9, 1.5, boxstyle="round,pad=0.1", 
                             facecolor='#F0F0F0', edgecolor='black', linewidth=1)
    ax.add_patch(math_box)
    
    math_text = """Key Equations:
Monte Carlo: E[f(x)] ≈ (1/N) Σ f(x, θᵢ) where θᵢ ~ q(θ)
Bayesian: P(θ|D) = P(D|θ)P(θ) / P(D)
Uncertainty: σ² = E[f²] - E[f]² (epistemic) + σ²ₙₒᵢₛₑ (aleatoric)"""
    
    ax.text(5, 1.25, math_text, ha='center', va='center', fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    """Generate Monte Carlo algorithm flowchart"""
    
    print("Generating Monte Carlo Solar Flare Model Algorithm Flowchart...")
    
    fig = create_monte_carlo_flowchart()
    fig.savefig("monte_carlo_algorithm_flowchart.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print("SUCCESS: Monte Carlo algorithm flowchart saved: monte_carlo_algorithm_flowchart.png")

if __name__ == "__main__":
    main()
