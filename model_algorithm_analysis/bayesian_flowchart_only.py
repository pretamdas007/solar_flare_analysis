"""
Simple Bayesian Solar Flare Model - Professional Algorithm Flowchart
Simplified version generating only the main algorithm flowchart
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import seaborn as sns

# Set style
plt.style.use('default')
sns.set_palette("Set1")

def create_bayesian_flowchart():
    """Create professional Bayesian algorithm flowchart"""
    
    fig, ax = plt.subplots(figsize=(14, 18))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 18)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#E8F4FD',
        'preprocessing': '#B8E6B8', 
        'bayesian': '#DDA0DD',
        'physics': '#98FB98',
        'monte_carlo': '#FF6B6B',
        'output': '#FFA07A',
        'uncertainty': '#20B2AA'
    }
    
    # Title
    ax.text(5, 17.5, 'Simple Bayesian Solar Flare Model Algorithm', 
            fontsize=18, fontweight='bold', ha='center')
    ax.text(5, 17, 'Physics-Informed Bayesian Neural Network', 
            fontsize=14, ha='center', style='italic')
    
    # Step 1: Input Data
    step1 = FancyBboxPatch((3.5, 15.5), 3, 1, boxstyle="round,pad=0.1", 
                          facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(step1)
    ax.text(5, 16, 'XRS Time Series Data\n(XRSA, XRSB channels)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 2: Physics-Based Generation
    step2 = FancyBboxPatch((3.5, 14), 3, 1.5, boxstyle="round,pad=0.1", 
                          facecolor=colors['physics'], edgecolor='black', linewidth=2)
    ax.add_patch(step2)
    ax.text(5, 14.75, 'Physics-Based Data\nExponential Profiles\nSynthetic Flares', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 3: Data Preprocessing
    step3 = FancyBboxPatch((3.5, 12), 3, 1, boxstyle="round,pad=0.1", 
                          facecolor=colors['preprocessing'], edgecolor='black', linewidth=2)
    ax.add_patch(step3)
    ax.text(5, 12.5, 'Data Preprocessing\nRobust Scaling', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 4: Bayesian Neural Network
    step4 = FancyBboxPatch((3.5, 10.5), 3, 1.5, boxstyle="round,pad=0.1", 
                          facecolor=colors['bayesian'], edgecolor='black', linewidth=2)
    ax.add_patch(step4)
    ax.text(5, 11.25, 'Bayesian Neural Network\nCNN + Dense Layers\nDropout for Uncertainty', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 5: Monte Carlo Inference
    step5 = FancyBboxPatch((3.5, 8.5), 3, 1.5, boxstyle="round,pad=0.1", 
                          facecolor=colors['monte_carlo'], edgecolor='black', linewidth=2)
    ax.add_patch(step5)
    ax.text(5, 9.25, 'Monte Carlo Inference\nMultiple Forward Passes\nActive Dropout', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 6: Parameter Estimation
    step6 = FancyBboxPatch((1, 6.5), 8, 1.5, boxstyle="round,pad=0.1", 
                          facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(step6)
    ax.text(5, 7.25, 'Flare Parameter Estimation\nAmplitude • Peak Position • Rise Time • Decay Time • Background', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Step 7: Uncertainty Quantification
    step7 = FancyBboxPatch((3.5, 4.5), 3, 1.5, boxstyle="round,pad=0.1", 
                          facecolor=colors['uncertainty'], edgecolor='black', linewidth=2)
    ax.add_patch(step7)
    ax.text(5, 5.25, 'Uncertainty Quantification\nConfidence Intervals\nPredictive Distributions', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 8: Nanoflare Detection
    step8a = FancyBboxPatch((1, 2.5), 3.5, 1, boxstyle="round,pad=0.1", 
                           facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(step8a)
    ax.text(2.75, 3, 'Nanoflare Detection\nAmplitude Thresholding', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Step 9: Results
    step8b = FancyBboxPatch((5.5, 2.5), 3.5, 1, boxstyle="round,pad=0.1", 
                           facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(step8b)
    ax.text(7.25, 3, 'Analysis Results\nParameter Distributions', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Add arrows
    arrows = [
        ((5, 15.5), (5, 15.5)),     # Input to physics
        ((5, 14), (5, 13)),         # Physics to preprocessing
        ((5, 12), (5, 11.8)),       # Preprocessing to Bayesian
        ((5, 10.5), (5, 10)),       # Bayesian to Monte Carlo
        ((5, 8.5), (5, 8)),         # Monte Carlo to parameters
        ((5, 6.5), (5, 6)),         # Parameters to uncertainty
        ((5, 4.5), (2.75, 3.5)),    # Uncertainty to nanoflare
        ((5, 4.5), (7.25, 3.5))     # Uncertainty to results
    ]
    
    for (start, end) in arrows:
        arrow = ConnectionPatch(start, end, "data", "data", 
                              arrowstyle="-|>", shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc="black")
        ax.add_patch(arrow)
    
    # Mathematical formulation
    math_box = FancyBboxPatch((0.5, 0.2), 9, 1.8, boxstyle="round,pad=0.1", 
                             facecolor='#F0F0F0', edgecolor='black', linewidth=1)
    ax.add_patch(math_box)
    
    math_text = """Key Equations:
Flare Profile: f(t) = A × exp(-(t-t₀)/τ) for t > t₀ (exponential decay)
Bayesian Inference: P(θ|D) = P(D|θ)P(θ) / P(D)
Monte Carlo: E[f(x)] ≈ (1/N) Σ f(x, θᵢ) where θᵢ ~ q(θ)
Parameters: [Amplitude, Peak_time, Rise_time, Decay_time, Background]
Nanoflare Threshold: A < 2×10⁻⁹ W/m²"""
    
    ax.text(5, 1.1, math_text, ha='center', va='center', fontsize=9, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    """Generate Bayesian algorithm flowchart"""
    
    print("Generating Simple Bayesian Solar Flare Model Algorithm Flowchart...")
    
    fig = create_bayesian_flowchart()
    fig.savefig("bayesian_algorithm_flowchart.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print("SUCCESS: Bayesian algorithm flowchart saved: bayesian_algorithm_flowchart.png")

if __name__ == "__main__":
    main()
