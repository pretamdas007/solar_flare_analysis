"""
Contrastive Learning Solar Flare Model - Professional Algorithm Flowchart
Simplified version generating only the main algorithm flowchart
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import seaborn as sns

# Set style
plt.style.use('default')
sns.set_palette("viridis")

def create_contrastive_learning_flowchart():
    """Create professional contrastive learning algorithm flowchart"""
    
    fig, ax = plt.subplots(figsize=(14, 18))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 18)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#E8F4FD',
        'augmentation': '#98FB98', 
        'encoder': '#FFD700',
        'projection': '#FF6B6B',
        'contrastive': '#9370DB',
        'classifier': '#FFA07A',
        'output': '#20B2AA'
    }
    
    # Title
    ax.text(5, 17.5, 'Self-Supervised Contrastive Learning Solar Flare Model', 
            fontsize=16, fontweight='bold', ha='center')
    ax.text(5, 17, 'SimCLR-based Representation Learning', 
            fontsize=14, ha='center', style='italic')
    
    # Step 1: Input Data
    step1 = FancyBboxPatch((3.5, 15.5), 3, 1, boxstyle="round,pad=0.1", 
                          facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(step1)
    ax.text(5, 16, 'XRS Time Series Data\nOriginal Sequences', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 2: Data Augmentation (Two branches)
    aug1 = FancyBboxPatch((1.5, 13.5), 2.5, 1.5, boxstyle="round,pad=0.1", 
                         facecolor=colors['augmentation'], edgecolor='black', linewidth=2)
    ax.add_patch(aug1)
    ax.text(2.75, 14.25, 'Augmentation 1\n• Noise\n• Time masking', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    aug2 = FancyBboxPatch((6, 13.5), 2.5, 1.5, boxstyle="round,pad=0.1", 
                         facecolor=colors['augmentation'], edgecolor='black', linewidth=2)
    ax.add_patch(aug2)
    ax.text(7.25, 14.25, 'Augmentation 2\n• Scaling\n• Time shifting', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Step 3: Encoder Network (Two branches)
    enc1 = FancyBboxPatch((1.5, 11.5), 2.5, 1.5, boxstyle="round,pad=0.1", 
                         facecolor=colors['encoder'], edgecolor='black', linewidth=2)
    ax.add_patch(enc1)
    ax.text(2.75, 12.25, 'Shared Encoder f(·)\n• Conv1D\n• Global pooling', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    enc2 = FancyBboxPatch((6, 11.5), 2.5, 1.5, boxstyle="round,pad=0.1", 
                         facecolor=colors['encoder'], edgecolor='black', linewidth=2)
    ax.add_patch(enc2)
    ax.text(7.25, 12.25, 'Shared Encoder f(·)\n• Same weights\n• Representation', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Step 4: Projection Head (Two branches)
    proj1 = FancyBboxPatch((1.5, 9.5), 2.5, 1.5, boxstyle="round,pad=0.1", 
                          facecolor=colors['projection'], edgecolor='black', linewidth=2)
    ax.add_patch(proj1)
    ax.text(2.75, 10.25, 'Projection g(·)\nz₁ = g(f(x₁))\nDim: 128', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    proj2 = FancyBboxPatch((6, 9.5), 2.5, 1.5, boxstyle="round,pad=0.1", 
                          facecolor=colors['projection'], edgecolor='black', linewidth=2)
    ax.add_patch(proj2)
    ax.text(7.25, 10.25, 'Projection g(·)\nz₂ = g(f(x₂))\nDim: 128', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Step 5: Contrastive Loss
    loss_box = FancyBboxPatch((3, 7.5), 4, 1.5, boxstyle="round,pad=0.1", 
                             facecolor=colors['contrastive'], edgecolor='black', linewidth=2)
    ax.add_patch(loss_box)
    ax.text(5, 8.25, 'NT-Xent Contrastive Loss\nMaximize Agreement\nBetween Positive Pairs', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 6: Fine-tuning Phase
    finetune_box = FancyBboxPatch((2, 5.5), 6, 1.5, boxstyle="round,pad=0.1", 
                                 facecolor=colors['classifier'], edgecolor='black', linewidth=2)
    ax.add_patch(finetune_box)
    ax.text(5, 6.25, 'Fine-tuning Phase\nFreeze Encoder + Add Classifier\nSupervised Learning', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Step 7: Final Predictions
    output_box = FancyBboxPatch((3, 3.5), 4, 1.5, boxstyle="round,pad=0.1", 
                               facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 4.25, 'Solar Flare Classification\nClass Predictions\nConfidence Scores', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Add arrows
    arrows = [
        # From input to augmentations
        ((5, 15.5), (2.75, 15)),
        ((5, 15.5), (7.25, 15)),
        # From augmentations to encoders
        ((2.75, 13.5), (2.75, 13)),
        ((7.25, 13.5), (7.25, 13)),
        # From encoders to projections
        ((2.75, 11.5), (2.75, 11)),
        ((7.25, 11.5), (7.25, 11)),
        # From projections to loss
        ((2.75, 9.5), (4, 9)),
        ((7.25, 9.5), (6, 9)),
        # From loss to fine-tuning
        ((5, 7.5), (5, 7)),
        # From fine-tuning to output
        ((5, 5.5), (5, 5))
    ]
    
    for (start, end) in arrows:
        arrow = ConnectionPatch(start, end, "data", "data", 
                              arrowstyle="-|>", shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc="black")
        ax.add_patch(arrow)
    
    # Mathematical formulation
    math_box = FancyBboxPatch((0.5, 0.5), 9, 2.5, boxstyle="round,pad=0.1", 
                             facecolor='#F0F0F0', edgecolor='black', linewidth=1)
    ax.add_patch(math_box)
    
    math_text = """Key Equations:
Similarity: sim(z₁, z₂) = z₁ᵀz₂ / (||z₁|| ||z₂||)
NT-Xent Loss: ℒᵢⱼ = -log(exp(sim(zᵢ,zⱼ)/τ) / Σₖ exp(sim(zᵢ,zₖ)/τ))
Total Loss: ℒ = (1/2N) Σᵢ [ℒ(2i-1,2i) + ℒ(2i,2i-1)]
Temperature: τ controls concentration (typically 0.1-0.5)
Two-Stage Training: (1) Unsupervised pretraining (2) Supervised fine-tuning"""
    
    ax.text(5, 1.75, math_text, ha='center', va='center', fontsize=9, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    """Generate Contrastive Learning algorithm flowchart"""
    
    print("Generating Contrastive Learning Solar Flare Model Algorithm Flowchart...")
    
    fig = create_contrastive_learning_flowchart()
    fig.savefig("contrastive_learning_algorithm_flowchart.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print("SUCCESS: Contrastive Learning algorithm flowchart saved: contrastive_learning_algorithm_flowchart.png")

if __name__ == "__main__":
    main()
