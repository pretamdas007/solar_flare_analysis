"""
Transformer Solar Flare Model - Professional Algorithm Flowchart
Simplified version generating only the main algorithm flowchart
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
plt.style.use('default')
sns.set_palette("tab10")

def create_transformer_flowchart():
    """Create professional transformer algorithm flowchart"""
    
    fig, ax = plt.subplots(figsize=(14, 18))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 18)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#E8F4FD',
        'embedding': '#B8E6B8', 
        'attention': '#FFD700',
        'transformer': '#FF6B6B',
        'output': '#FFA07A',
        'processing': '#DDA0DD'
    }
    
    # Title
    ax.text(5, 17.5, 'Transformer Solar Flare Model Algorithm', 
            fontsize=18, fontweight='bold', ha='center')
    ax.text(5, 17, 'Multi-Head Self-Attention for Sequence Modeling', 
            fontsize=14, ha='center', style='italic')
    
    # Step 1: Input Data
    step1 = FancyBboxPatch((3.5, 15.5), 3, 1, boxstyle="round,pad=0.1", 
                          facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(step1)
    ax.text(5, 16, 'XRS Time Series Input\n(Sequence Length × Features)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 2: Input Embedding
    step2 = FancyBboxPatch((3.5, 14), 3, 1, boxstyle="round,pad=0.1", 
                          facecolor=colors['embedding'], edgecolor='black', linewidth=2)
    ax.add_patch(step2)
    ax.text(5, 14.5, 'Input Embedding\n+ Positional Encoding', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 3: Multi-Head Attention
    step3 = FancyBboxPatch((3.5, 12.5), 3, 1, boxstyle="round,pad=0.1", 
                          facecolor=colors['attention'], edgecolor='black', linewidth=2)
    ax.add_patch(step3)
    ax.text(5, 13, 'Multi-Head Self-Attention\nQ, K, V Transformations', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 4: Add & Norm
    step4 = FancyBboxPatch((3.5, 11), 3, 1, boxstyle="round,pad=0.1", 
                          facecolor=colors['processing'], edgecolor='black', linewidth=2)
    ax.add_patch(step4)
    ax.text(5, 11.5, 'Add & Normalize\nResidual Connection', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 5: Feed Forward
    step5 = FancyBboxPatch((3.5, 9.5), 3, 1, boxstyle="round,pad=0.1", 
                          facecolor=colors['transformer'], edgecolor='black', linewidth=2)
    ax.add_patch(step5)
    ax.text(5, 10, 'Feed Forward Network\nReLU Activation', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 6: Add & Norm 2
    step6 = FancyBboxPatch((3.5, 8), 3, 1, boxstyle="round,pad=0.1", 
                          facecolor=colors['processing'], edgecolor='black', linewidth=2)
    ax.add_patch(step6)
    ax.text(5, 8.5, 'Add & Normalize\nSecond Residual', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 7: Transformer Layers (N×)
    step7 = FancyBboxPatch((1, 6.5), 8, 1, boxstyle="round,pad=0.1", 
                          facecolor=colors['transformer'], edgecolor='black', linewidth=2)
    ax.add_patch(step7)
    ax.text(5, 7, '× N Transformer Layers (Encoder Stack)', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Step 8: Classification Head
    step8 = FancyBboxPatch((1, 5), 3.5, 1, boxstyle="round,pad=0.1", 
                          facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(step8)
    ax.text(2.75, 5.5, 'Classification Head\nFlare Classes (A,B,C,M,X)', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Step 9: Regression Head
    step9 = FancyBboxPatch((5.5, 5), 3.5, 1, boxstyle="round,pad=0.1", 
                          facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(step9)
    ax.text(7.25, 5.5, 'Regression Head\nFlare Intensity Prediction', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Final Output
    step10 = FancyBboxPatch((3.5, 3.5), 3, 1, boxstyle="round,pad=0.1", 
                           facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(step10)
    ax.text(5, 4, 'Multi-Task Predictions\nClass + Intensity', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Add arrows
    arrows = [
        ((5, 15.5), (5, 15)),    # Input to embedding
        ((5, 14), (5, 13.5)),    # Embedding to attention
        ((5, 12.5), (5, 12)),    # Attention to add&norm
        ((5, 11), (5, 10.5)),    # Add&norm to FFN
        ((5, 9.5), (5, 9)),      # FFN to add&norm2
        ((5, 8), (5, 7.5)),      # To transformer layers
        ((5, 6.5), (2.75, 6)),   # To classification
        ((5, 6.5), (7.25, 6)),   # To regression
        ((2.75, 5), (4, 4.5)),   # Classification to output
        ((7.25, 5), (6, 4.5))    # Regression to output
    ]
    
    for (start, end) in arrows:
        arrow = ConnectionPatch(start, end, "data", "data", 
                              arrowstyle="-|>", shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc="black")
        ax.add_patch(arrow)
    
    # Add mathematical formulation
    math_box = FancyBboxPatch((0.5, 1), 9, 2, boxstyle="round,pad=0.1", 
                             facecolor='#F0F0F0', edgecolor='black', linewidth=1)
    ax.add_patch(math_box)
    
    math_text = """Key Equations:
Self-Attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
Multi-Head: MultiHead(Q,K,V) = Concat(head₁,...,head_h)W^O
Layer Norm: LayerNorm(x) = γ(x-μ)/σ + β
Feed Forward: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂"""
    
    ax.text(5, 2, math_text, ha='center', va='center', fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    """Generate Transformer algorithm flowchart"""
    
    print("Generating Transformer Solar Flare Model Algorithm Flowchart...")
    
    fig = create_transformer_flowchart()
    fig.savefig("transformer_algorithm_flowchart.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print("SUCCESS: Transformer algorithm flowchart saved: transformer_algorithm_flowchart.png")

if __name__ == "__main__":
    main()
