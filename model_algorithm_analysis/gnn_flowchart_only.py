"""
Graph Neural Network Solar Flare Model - Professional Algorithm Flowchart
Simplified version generating only the main algorithm flowchart
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import seaborn as sns

# Set style
plt.style.use('default')
sns.set_palette("Set2")

def create_gnn_flowchart():
    """Create professional GNN algorithm flowchart"""
    
    fig, ax = plt.subplots(figsize=(14, 18))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 18)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#E8F4FD',
        'graph': '#98FB98', 
        'attention': '#FFD700',
        'gnn': '#FF6B6B',
        'aggregation': '#9370DB',
        'output': '#FFA07A'
    }
    
    # Title
    ax.text(5, 17.5, 'Graph Neural Network Solar Flare Model Algorithm', 
            fontsize=18, fontweight='bold', ha='center')
    ax.text(5, 17, 'Modeling Temporal-Spatial Relationships in Solar Data', 
            fontsize=14, ha='center', style='italic')
    
    # Step 1: Time Series Input
    step1 = FancyBboxPatch((3.5, 15.5), 3, 1, boxstyle="round,pad=0.1", 
                          facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(step1)
    ax.text(5, 16, 'XRS Time Series Data\nSequence: T × F', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 2: Graph Construction
    step2 = FancyBboxPatch((3.5, 14), 3, 1.5, boxstyle="round,pad=0.1", 
                          facecolor=colors['graph'], edgecolor='black', linewidth=2)
    ax.add_patch(step2)
    ax.text(5, 14.75, 'Graph Construction\nNodes: Time Steps\nEdges: k-NN Similarity', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 3: Node Features
    step3 = FancyBboxPatch((3.5, 12), 3, 1, boxstyle="round,pad=0.1", 
                          facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(step3)
    ax.text(5, 12.5, 'Node Feature Matrix\nH⁽⁰⁾ ∈ ℝᴺˣᶠ', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 4: Graph Attention Layer 1
    step4 = FancyBboxPatch((3.5, 10.5), 3, 1, boxstyle="round,pad=0.1", 
                          facecolor=colors['attention'], edgecolor='black', linewidth=2)
    ax.add_patch(step4)
    ax.text(5, 11, 'Graph Attention Layer 1\nMulti-Head Attention', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 5: Message Passing
    step5 = FancyBboxPatch((1, 9), 3.5, 1, boxstyle="round,pad=0.1", 
                          facecolor=colors['gnn'], edgecolor='black', linewidth=2)
    ax.add_patch(step5)
    ax.text(2.75, 9.5, 'Message Passing\nAggregate Neighbors', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 6: Feature Update
    step6 = FancyBboxPatch((5.5, 9), 3.5, 1, boxstyle="round,pad=0.1", 
                          facecolor=colors['gnn'], edgecolor='black', linewidth=2)
    ax.add_patch(step6)
    ax.text(7.25, 9.5, 'Feature Update\nh⁽ˡ⁺¹⁾ = σ(W h⁽ˡ⁾)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 7: Graph Attention Layer 2
    step7 = FancyBboxPatch((3.5, 7.5), 3, 1, boxstyle="round,pad=0.1", 
                          facecolor=colors['attention'], edgecolor='black', linewidth=2)
    ax.add_patch(step7)
    ax.text(5, 8, 'Graph Attention Layer 2\nDeeper Representations', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 8: Graph Pooling
    step8 = FancyBboxPatch((3.5, 6), 3, 1, boxstyle="round,pad=0.1", 
                          facecolor=colors['aggregation'], edgecolor='black', linewidth=2)
    ax.add_patch(step8)
    ax.text(5, 6.5, 'Graph Pooling\nMean/Max/Attention', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 9: Multi-task Outputs
    step9a = FancyBboxPatch((1.5, 4), 3, 1, boxstyle="round,pad=0.1", 
                           facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(step9a)
    ax.text(3, 4.5, 'Classification\nFlare Classes', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    step9b = FancyBboxPatch((5.5, 4), 3, 1, boxstyle="round,pad=0.1", 
                           facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(step9b)
    ax.text(7, 4.5, 'Regression\nEnergy Estimation', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 10: Final Predictions
    step10 = FancyBboxPatch((3.5, 2), 3, 1, boxstyle="round,pad=0.1", 
                           facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(step10)
    ax.text(5, 2.5, 'Combined Predictions\nClass + Energy', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Add arrows
    arrows = [
        ((5, 15.5), (5, 15.5)),     # Input to graph construction
        ((5, 14), (5, 13)),         # Graph to node features
        ((5, 12), (5, 11.5)),       # Node features to GAT1
        ((5, 10.5), (2.75, 10)),    # GAT1 to message passing
        ((5, 10.5), (7.25, 10)),    # GAT1 to feature update
        ((2.75, 9), (4, 8.5)),      # Message passing to GAT2
        ((7.25, 9), (6, 8.5)),      # Feature update to GAT2
        ((5, 7.5), (5, 7)),         # GAT2 to pooling
        ((5, 6), (3, 5)),           # Pooling to classification
        ((5, 6), (7, 5)),           # Pooling to regression
        ((3, 4), (4, 3)),           # Classification to final
        ((7, 4), (6, 3))            # Regression to final
    ]
    
    for (start, end) in arrows:
        arrow = ConnectionPatch(start, end, "data", "data", 
                              arrowstyle="-|>", shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc="black")
        ax.add_patch(arrow)
    
    # Mathematical formulation
    math_box = FancyBboxPatch((0.5, 0.2), 9, 1.5, boxstyle="round,pad=0.1", 
                             facecolor='#F0F0F0', edgecolor='black', linewidth=1)
    ax.add_patch(math_box)
    
    math_text = """Key Equations:
Attention: αᵢⱼ = softmax(LeakyReLU(aᵀ[Whᵢ ∥ Whⱼ]))
Update: hᵢ⁽ˡ⁺¹⁾ = σ(Σⱼ∈𝒩ᵢ αᵢⱼ⁽ˡ⁾ W⁽ˡ⁾hⱼ⁽ˡ⁾)
Pooling: h_graph = POOL({hᵢ⁽ᴸ⁾ : i ∈ V})"""
    
    ax.text(5, 0.95, math_text, ha='center', va='center', fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    """Generate GNN algorithm flowchart"""
    
    print("Generating Graph Neural Network Solar Flare Model Algorithm Flowchart...")
    
    fig = create_gnn_flowchart()
    fig.savefig("gnn_algorithm_flowchart.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print("SUCCESS: GNN algorithm flowchart saved: gnn_algorithm_flowchart.png")

if __name__ == "__main__":
    main()
