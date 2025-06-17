"""
Graph Neural Network for Solar Flare Analysis
Models complex relationships between different solar parameters
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import kneighbors_graph
from typing import Optional, Tuple, List
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')


class GraphAttentionLayer(layers.Layer):
    """
    Graph Attention Layer implementation
    """
    
    def __init__(self, units, num_heads=8, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate        
        self.W = layers.Dense(units * num_heads, use_bias=False)
        self.a = layers.Dense(2 * units * num_heads, use_bias=False)
        self.dropout = layers.Dropout(dropout_rate)
        self.leaky_relu = layers.LeakyReLU(alpha=0.2)
        
    def call(self, inputs, adjacency_matrix, training=None):
        """
        Forward pass through graph attention layer with memory optimization
        """
        batch_size = tf.shape(inputs)[0]
        num_nodes = tf.shape(inputs)[1]
        
        # Linear transformation
        h = self.W(inputs)  # [batch, num_nodes, units * num_heads]
        h = tf.reshape(h, [batch_size, num_nodes, self.num_heads, self.units])
        
        # More memory-efficient attention computation
        # Process one head at a time to reduce memory usage
        attention_outputs = []
        
        for head in range(self.num_heads):
            h_head = h[:, :, head, :]  # [batch, num_nodes, units]
            
            # Compute attention for this head
            h_i = tf.expand_dims(h_head, axis=2)  # [batch, num_nodes, 1, units]
            h_j = tf.expand_dims(h_head, axis=1)  # [batch, 1, num_nodes, units]
            
            # Concatenate for attention computation
            concat = tf.concat([
                tf.tile(h_i, [1, 1, num_nodes, 1]),
                tf.tile(h_j, [1, num_nodes, 1, 1])
            ], axis=-1)  # [batch, num_nodes, num_nodes, 2*units]
            
            # Compute attention scores
            e = self.a(concat)  # [batch, num_nodes, num_nodes, 2*units]
            e = tf.reduce_sum(e, axis=-1)  # [batch, num_nodes, num_nodes]
            e = self.leaky_relu(e)
            
            # Apply adjacency mask
            e = tf.where(adjacency_matrix > 0, e, -1e9)
            
            # Softmax attention weights
            alpha = tf.nn.softmax(e, axis=2)
            alpha = self.dropout(alpha, training=training)
            
            # Apply attention to node features
            h_prime = tf.einsum('bij,bjk->bik', alpha, h_head)
            attention_outputs.append(h_prime)
        
        # Combine heads
        if self.num_heads > 1:
            h_prime = tf.reduce_mean(tf.stack(attention_outputs, axis=-1), axis=-1)
        else:
            h_prime = attention_outputs[0]
        
        return h_prime


class GraphNeuralFlareModel:
    """    Graph Neural Network for modeling complex solar flare relationships
    """
    
    def __init__(self,
                 sequence_length: int = 128,
                 n_features: int = 2,
                 n_classes: int = 6,
                 hidden_units: int = 32,  # Reduced from 64
                 num_gat_layers: int = 2,  # Reduced from 3
                 num_heads: int = 4,      # Reduced from 8
                 k_neighbors: int = 3):   # Reduced from 5
        """
        Initialize Graph Neural Network model
        
        Parameters
        ----------
        sequence_length : int
            Length of input sequences
        n_features : int
            Number of input features
        n_classes : int
            Number of flare classes
        hidden_units : int
            Hidden layer dimensions
        num_gat_layers : int
            Number of Graph Attention layers        num_heads : int
            Number of attention heads        k_neighbors : int
            Number of neighbors for graph construction
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_units = hidden_units
        self.num_gat_layers = num_gat_layers
        self.num_heads = num_heads
        self.k_neighbors = k_neighbors
        
        self.model = None
        self.history = None
        self.scaler_X = RobustScaler()
        
    def construct_graph(self, X):
        """
        Construct k-NN graph from input data with memory optimization
        """
        batch_size = X.shape[0]
        adjacency_matrices = []
        
        # Use smaller k for memory efficiency but keep original sequence length
        k_neighbors = min(self.k_neighbors, self.sequence_length//8, 5)
        
        for i in range(batch_size):
            # Flatten sequence for distance computation
            sample = X[i].reshape(self.sequence_length, -1)
            
            # For very long sequences, create a simpler adjacency matrix
            if self.sequence_length > 64:
                # Create a sparse adjacency matrix using only local connections
                adj_matrix = np.zeros((self.sequence_length, self.sequence_length))
                
                # Connect each node to its k nearest temporal neighbors
                for j in range(self.sequence_length):
                    # Connect to previous and next neighbors
                    for offset in range(1, min(k_neighbors + 1, self.sequence_length//2)):
                        if j - offset >= 0:
                            adj_matrix[j, j - offset] = 1
                        if j + offset < self.sequence_length:
                            adj_matrix[j, j + offset] = 1
                    # Always connect to self
                    adj_matrix[j, j] = 1
            else:
                # For shorter sequences, use full k-NN graph
                adj_matrix = kneighbors_graph(
                    sample, n_neighbors=min(k_neighbors, self.sequence_length//2),
                    mode='connectivity', include_self=True
                ).toarray()
            
            adjacency_matrices.append(adj_matrix)
        
        return np.array(adjacency_matrices, dtype=np.float32)
    
    def build_model(self) -> keras.Model:
        """
        Build the Graph Neural Network model
        """
        # Inputs
        node_features = layers.Input(
            shape=(self.sequence_length, self.n_features),
            name='node_features'
        )
        adjacency_input = layers.Input(
            shape=(self.sequence_length, self.sequence_length),
            name='adjacency_matrix'
        )
          # Initial node embedding with reduced dimensions
        x = layers.Dense(self.hidden_units//2, activation='relu')(node_features)
        
        # Graph Attention layers with reduced complexity
        for i in range(self.num_gat_layers):
            x = GraphAttentionLayer(
                units=self.hidden_units//2,  # Reduced units
                num_heads=max(2, self.num_heads//2),  # Reduced heads
                dropout_rate=0.2,  # Increased dropout
                name=f'gat_layer_{i}'
            )(x, adjacency_input)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        
        # Global graph pooling
        graph_embedding = layers.GlobalAveragePooling1D()(x)
        
        # Final classification layers with reduced dimensions
        x = layers.Dense(64, activation='relu')(graph_embedding)  # Reduced from 256
        x = layers.Dropout(0.4)(x)  # Increased dropout
        x = layers.Dense(32, activation='relu')(x)  # Reduced from 128
        x = layers.Dropout(0.3)(x)
        
        # Multi-task outputs
        flare_class = layers.Dense(
            self.n_classes, activation='softmax', name='flare_class'
        )(x)
        flare_energy = layers.Dense(
            1, activation='linear', name='flare_energy'
        )(x)
        
        model = keras.Model(
            inputs=[node_features, adjacency_input],
            outputs=[flare_class, flare_energy]
        )
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss={
                'flare_class': 'sparse_categorical_crossentropy',
                'flare_energy': 'mse'
            },
            loss_weights={'flare_class': 1.0, 'flare_energy': 0.3},
            metrics={
                'flare_class': ['accuracy'],
                'flare_energy': ['mae']
            }
        )
        
        self.model = model
        return model
    
    def preprocess_data(self, X, y_class=None, y_energy=None):
        """
        Preprocess data for graph neural network
        """
        # Scale features
        X_scaled = self.scaler_X.fit_transform(
            X.reshape(-1, self.n_features)
        ).reshape(X.shape)
        
        # Construct adjacency matrices
        adjacency_matrices = self.construct_graph(X_scaled)
        
        if y_class is not None and y_energy is not None:
            return [X_scaled, adjacency_matrices], [y_class, y_energy]
        else:
            return [X_scaled, adjacency_matrices]
    
    def train(self, X_train, y_train_class, y_train_energy,
              X_val, y_val_class, y_val_energy,
              epochs=100, batch_size=32, verbose=1):
        """
        Train the graph neural network
        """
        if self.model is None:
            self.build_model()
        
        # Preprocess data
        X_train_processed, y_train_processed = self.preprocess_data(
            X_train, y_train_class, y_train_energy
        )
        X_val_processed, y_val_processed = self.preprocess_data(
            X_val, y_val_class, y_val_energy
        )
        
        # Callbacks
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_graph_model.h5', monitor='val_loss', save_best_only=True
            )
        ]
        
        # Train
        self.history = self.model.fit(
            X_train_processed, y_train_processed,
            validation_data=(X_val_processed, y_val_processed),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions using the trained model
        """
        X_processed = self.preprocess_data(X)
        return self.model.predict(X_processed)
    def visualize_graph(self, X_sample, sample_idx=0, save_path=None):
        """
        Enhanced graph visualization with professional seaborn aesthetics
        """
        adjacency = self.construct_graph(X_sample[sample_idx:sample_idx+1])[0]
        
        # Set professional seaborn styling
        plt.style.use('seaborn-v0_8')
        sns.set_theme(style="whitegrid", palette="viridis", font_scale=1.1)
        sns.set_context("paper", rc={"figure.dpi": 300})
        
        # Create comprehensive graph analysis dashboard
        fig = plt.figure(figsize=(20, 12), facecolor='white')
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
        
        # 1. Main Graph Visualization
        ax1 = fig.add_subplot(gs[0, :2])
        G = nx.from_numpy_array(adjacency)
        
        # Enhanced graph layout
        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
        
        # Calculate node properties for enhanced visualization
        node_degrees = dict(G.degree())
        node_sizes = [300 + degree * 50 for degree in node_degrees.values()]
        edge_weights = [adjacency[u, v] * 5 for u, v in G.edges()]
        
        # Draw with professional styling
        nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=list(node_degrees.values()),
                              node_size=node_sizes, cmap='viridis', alpha=0.8,
                              edgecolors='black', linewidths=0.5)
        nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.6, width=edge_weights,
                              edge_color='gray', style='-')
        
        # Add node labels
        nx.draw_networkx_labels(G, pos, ax=ax1, font_size=8, font_weight='bold')
        
        ax1.set_title('🌐 Graph Neural Network Structure', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.axis('off')
        
        # 2. Adjacency Matrix Heatmap
        ax2 = fig.add_subplot(gs[0, 2])
        sns.heatmap(adjacency, ax=ax2, cmap='viridis', cbar=True,
                   square=True, linewidths=0.1, linecolor='white',
                   cbar_kws={'label': 'Connection Strength', 'shrink': 0.8})
        ax2.set_title('Adjacency Matrix', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Node Index', fontsize=11)
        ax2.set_ylabel('Node Index', fontsize=11)
        
        # 3. Node Degree Distribution
        ax3 = fig.add_subplot(gs[1, 0])
        degrees = list(node_degrees.values())
        sns.histplot(degrees, kde=True, ax=ax3, color='skyblue', alpha=0.7)
        ax3.set_title('Node Degree Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Node Degree', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # 4. Edge Weight Distribution
        ax4 = fig.add_subplot(gs[1, 1])
        edge_weights_dist = adjacency[adjacency > 0].flatten()
        sns.histplot(edge_weights_dist, kde=True, ax=ax4, color='coral', alpha=0.7)
        ax4.set_title('Edge Weight Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Edge Weight', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.grid(True, alpha=0.3)
        
        # 5. Graph Statistics
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        # Calculate graph metrics
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        density = nx.density(G)
        avg_clustering = nx.average_clustering(G)
        
        stats_text = f"""📊 GRAPH STATISTICS
        
🔢 Basic Metrics:
• Nodes: {n_nodes}
• Edges: {n_edges}
• Density: {density:.3f}
• Avg Clustering: {avg_clustering:.3f}

🎯 Connectivity:
• Max Degree: {max(degrees)}
• Min Degree: {min(degrees)}
• Avg Degree: {np.mean(degrees):.2f}

⚡ Structure:
• Connected: {'Yes' if nx.is_connected(G) else 'No'}
• Sample: {sample_idx}
        """
        
        ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen', alpha=0.9))
        
        fig.suptitle('🚀 Professional Graph Neural Network Analysis', 
                    fontsize=18, fontweight='bold', y=0.95,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightsteelblue', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def plot_training_history(self, history, save_path=None):
        """
        Enhanced training history visualization for GNN with seaborn
        """
        # Set professional seaborn styling
        plt.style.use('seaborn-v0_8')
        sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
        fig.suptitle('🎯 Graph Neural Network Training Dashboard', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        epochs = range(1, len(history.history['loss']) + 1)
        
        # 1. Multi-task Loss Evolution
        loss_data = []
        for epoch, loss in enumerate(history.history['loss'], 1):
            loss_data.append({'Epoch': epoch, 'Loss': loss, 'Type': 'Total Loss'})
        
        # Add specific task losses if available
        if 'classification_output_loss' in history.history:
            for epoch, loss in enumerate(history.history['classification_output_loss'], 1):
                loss_data.append({'Epoch': epoch, 'Loss': loss, 'Type': 'Classification'})
        
        if 'energy_output_loss' in history.history:
            for epoch, loss in enumerate(history.history['energy_output_loss'], 1):
                loss_data.append({'Epoch': epoch, 'Loss': loss, 'Type': 'Energy Regression'})
        
        loss_df = pd.DataFrame(loss_data)
        sns.lineplot(data=loss_df, x='Epoch', y='Loss', hue='Type', 
                    ax=axes[0,0], marker='o', linewidth=2.5, markersize=4)
        axes[0,0].set_title('Multi-task Training Loss', fontsize=14, fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_yscale('log')
        
        # 2. Validation Metrics
        val_data = []
        for metric in ['val_loss', 'val_classification_output_loss', 'val_energy_output_loss']:
            if metric in history.history:
                for epoch, value in enumerate(history.history[metric], 1):
                    val_data.append({'Epoch': epoch, 'Value': value, 'Metric': metric.replace('val_', '').replace('_', ' ').title()})
        
        if val_data:
            val_df = pd.DataFrame(val_data)
            sns.lineplot(data=val_df, x='Epoch', y='Value', hue='Metric', 
                        ax=axes[0,1], marker='s', linewidth=2.5, markersize=4)
            axes[0,1].set_title('Validation Metrics', fontsize=14, fontweight='bold')
            axes[0,1].set_yscale('log')
        else:
            axes[0,1].text(0.5, 0.5, 'Validation Metrics\nNot Available', ha='center', va='center',
                          transform=axes[0,1].transAxes, fontsize=12, fontweight='bold')
            axes[0,1].set_title('Validation Metrics', fontsize=14, fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Learning Rate and Accuracy
        if 'lr' in history.history:
            sns.lineplot(x=epochs, y=history.history['lr'], ax=axes[1,0], 
                        marker='d', linewidth=2.5, markersize=4, color='orange')
            axes[1,0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[1,0].set_yscale('log')
        else:
            # Show accuracy if available
            if 'classification_output_accuracy' in history.history:
                sns.lineplot(x=epochs, y=history.history['classification_output_accuracy'], 
                            ax=axes[1,0], marker='o', linewidth=2.5, markersize=4, color='green')
                axes[1,0].set_title('Classification Accuracy', fontsize=14, fontweight='bold')
            else:
                axes[1,0].text(0.5, 0.5, 'Learning Rate\nNot Tracked', ha='center', va='center',
                              transform=axes[1,0].transAxes, fontsize=12, fontweight='bold')
                axes[1,0].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Training Summary
        axes[1,1].axis('off')
        summary_text = f"""📊 GNN TRAINING SUMMARY
        
🏆 Final Metrics:
• Total Loss: {history.history['loss'][-1]:.4f}
• Classification Loss: {history.history.get('classification_output_loss', [0])[-1]:.4f}
• Energy Loss: {history.history.get('energy_output_loss', [0])[-1]:.4f}

🎯 Best Performance:
• Min Total Loss: {min(history.history['loss']):.4f}
• Total Epochs: {len(history.history['loss'])}

⚡ Model Configuration:
• Architecture: Graph Attention Network
• Hidden Units: {self.hidden_units}
• GAT Layers: {self.num_gat_layers}
• Attention Heads: {self.num_heads}
        """
        
        axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes,
                      fontsize=10, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.9))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def plot_graph_attention_analysis(self, X_sample, predictions, save_path=None):
        """
        Analyze graph attention patterns and node importance
        """
        # Set professional seaborn styling
        plt.style.use('seaborn-v0_8')
        sns.set_theme(style="whitegrid", palette="viridis", font_scale=1.1)
        
        fig = plt.figure(figsize=(20, 12), facecolor='white')
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
        
        # Get graph structure for first sample
        adjacency = self.construct_graph(X_sample[:1])[0]
        G = nx.from_numpy_array(adjacency)
        
        # 1. Node Importance Heatmap
        ax1 = fig.add_subplot(gs[0, :2])
        
        # Calculate node features importance
        node_features = X_sample[0]  # First sample
        feature_importance = np.std(node_features, axis=1)  # Variability as importance
        
        # Create heatmap
        importance_matrix = node_features.T
        sns.heatmap(importance_matrix, ax=ax1, cmap='viridis', cbar=True,
                   xticklabels=[f'Node {i}' for i in range(len(node_features))],
                   yticklabels=[f'Feature {i}' for i in range(node_features.shape[1])],
                   cbar_kws={'label': 'Feature Value'})
        ax1.set_title('🎯 Node Feature Importance Matrix', fontsize=16, fontweight='bold')
        
        # 2. Graph Centrality Analysis
        ax2 = fig.add_subplot(gs[0, 2])
        
        # Calculate centrality measures
        centrality_data = []
        betweenness = nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)
        degree = nx.degree_centrality(G)
        
        for node in G.nodes():
            centrality_data.append({
                'Node': node,
                'Betweenness': betweenness[node],
                'Closeness': closeness[node],
                'Degree': degree[node]
            })
        
        centrality_df = pd.DataFrame(centrality_data)
        centrality_melted = centrality_df.melt(id_vars=['Node'], var_name='Centrality', value_name='Score')
        
        sns.boxplot(data=centrality_melted, x='Centrality', y='Score', ax=ax2)
        ax2.set_title('Node Centrality Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Prediction Confidence by Node
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Assume predictions contain confidence scores
        node_predictions = predictions[0] if len(predictions.shape) > 1 else [predictions[0]] * len(G.nodes())
        pred_data = pd.DataFrame({
            'Node': list(G.nodes()),
            'Prediction': node_predictions[:len(G.nodes())],
            'Importance': feature_importance
        })
        
        sns.scatterplot(data=pred_data, x='Importance', y='Prediction', ax=ax3,
                       size='Prediction', sizes=(50, 200), alpha=0.7)
        ax3.set_title('Prediction vs Node Importance', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Graph Connectivity Analysis
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Analyze connectivity patterns
        connectivity_data = []
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            connectivity_data.append({
                'Node': node,
                'Neighbors': len(neighbors),
                'Weight_Sum': sum(adjacency[node, neighbor] for neighbor in neighbors)
            })
        
        conn_df = pd.DataFrame(connectivity_data)
        sns.scatterplot(data=conn_df, x='Neighbors', y='Weight_Sum', ax=ax4,
                       alpha=0.7, s=80)
        ax4.set_title('Node Connectivity Analysis', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Number of Neighbors')
        ax4.set_ylabel('Total Edge Weight')
        ax4.grid(True, alpha=0.3)
        
        # 5. Model Performance Summary
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        # Calculate graph statistics
        avg_prediction = np.mean(predictions)
        prediction_std = np.std(predictions)
        
        performance_text = f"""📊 GRAPH ANALYSIS SUMMARY
        
🎯 Prediction Statistics:
• Mean Prediction: {avg_prediction:.4f}
• Std Deviation: {prediction_std:.4f}
• Sample Count: {len(predictions)}

🌐 Graph Properties:
• Total Nodes: {G.number_of_nodes()}
• Total Edges: {G.number_of_edges()}
• Graph Density: {nx.density(G):.3f}
• Avg Clustering: {nx.average_clustering(G):.3f}

⚡ Feature Analysis:
• Feature Dimensions: {node_features.shape[1]}
• Max Importance: {max(feature_importance):.3f}
• Min Importance: {min(feature_importance):.3f}
        """
        
        ax5.text(0.05, 0.95, performance_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightcyan', alpha=0.9))
        
        fig.suptitle('🚀 Professional Graph Attention Analysis Dashboard', 
                    fontsize=18, fontweight='bold', y=0.95,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightsteelblue', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()


class HybridGraphTransformerModel:
    """
    Combines Graph Neural Networks with Transformer attention
    """
    
    def __init__(self,
                 sequence_length: int = 128,
                 n_features: int = 2,
                 n_classes: int = 6,
                 gnn_hidden_units: int = 64,
                 transformer_d_model: int = 128,
                 num_heads: int = 8):
        """
        Initialize hybrid Graph-Transformer model
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.gnn_hidden_units = gnn_hidden_units
        self.transformer_d_model = transformer_d_model
        self.num_heads = num_heads
        
        self.model = None
        self.scaler_X = RobustScaler()
    
    def build_model(self) -> keras.Model:
        """
        Build hybrid Graph-Transformer model
        """
        # Inputs
        node_features = layers.Input(
            shape=(self.sequence_length, self.n_features)
        )
        adjacency_input = layers.Input(
            shape=(self.sequence_length, self.sequence_length)
        )
        
        # Graph processing branch
        graph_x = layers.Dense(self.gnn_hidden_units, activation='relu')(node_features)
        graph_x = GraphAttentionLayer(
            units=self.gnn_hidden_units,
            num_heads=4,
            name='graph_attention'
        )(graph_x, adjacency_input)
        graph_features = layers.GlobalAveragePooling1D()(graph_x)
        
        # Transformer processing branch
        trans_x = layers.Dense(self.transformer_d_model)(node_features)
        
        # Positional encoding
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        positions = layers.Embedding(
            input_dim=self.sequence_length,
            output_dim=self.transformer_d_model
        )(positions)
        trans_x = trans_x + positions
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.transformer_d_model
        )(trans_x, trans_x)
        attention_output = layers.LayerNormalization()(trans_x + attention_output)
        
        # Feed-forward
        ffn_output = layers.Dense(256, activation="relu")(attention_output)
        ffn_output = layers.Dense(self.transformer_d_model)(ffn_output)
        trans_x = layers.LayerNormalization()(attention_output + ffn_output)
        
        trans_features = layers.GlobalAveragePooling1D()(trans_x)
        
        # Combine features
        combined = layers.Concatenate()([graph_features, trans_features])
        combined = layers.Dense(256, activation='relu')(combined)
        combined = layers.Dropout(0.3)(combined)
        
        # Output layers
        flare_class = layers.Dense(
            self.n_classes, activation='softmax', name='flare_class'
        )(combined)
        
        model = keras.Model(
            inputs=[node_features, adjacency_input],
            outputs=flare_class
        )
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def preprocess_data(self, X):
        """
        Preprocess data for hybrid model - create adjacency matrices
        """
        batch_size = X.shape[0]
        adjacency_matrices = []
        
        for i in range(batch_size):
            # Flatten sequence for distance computation
            sample = X[i].reshape(self.sequence_length, -1)
            
            # Compute k-NN graph with smaller k for memory efficiency
            k_neighbors = min(3, self.sequence_length//8)  # Reduced k for memory
            adj_matrix = kneighbors_graph(
                sample, n_neighbors=k_neighbors,
                mode='connectivity', include_self=True
            ).toarray()
            
            adjacency_matrices.append(adj_matrix)
        
        return [X, np.array(adjacency_matrices, dtype=np.float32)]
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=16, verbose=1):
        """
        Train the hybrid model
        """
        # Preprocess training data
        X_train_processed = self.preprocess_data(X_train)
        X_val_processed = self.preprocess_data(X_val)
        
        # Train model
        history = self.model.fit(
            X_train_processed, y_train,
            validation_data=(X_val_processed, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        return history

    def build_model(self) -> keras.Model:
        """
        Build hybrid Graph-Transformer model
        """
        # Inputs
        node_features = layers.Input(
            shape=(self.sequence_length, self.n_features)
        )
        adjacency_input = layers.Input(
            shape=(self.sequence_length, self.sequence_length)
        )
        
        # Graph processing branch
        graph_x = layers.Dense(self.gnn_hidden_units, activation='relu')(node_features)
        graph_x = GraphAttentionLayer(
            units=self.gnn_hidden_units,
            num_heads=4,
            name='graph_attention'
        )(graph_x, adjacency_input)
        graph_features = layers.GlobalAveragePooling1D()(graph_x)
        
        # Transformer processing branch
        trans_x = layers.Dense(self.transformer_d_model)(node_features)
        
        # Positional encoding
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        positions = layers.Embedding(
            input_dim=self.sequence_length,
            output_dim=self.transformer_d_model
        )(positions)
        trans_x = trans_x + positions
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.transformer_d_model
        )(trans_x, trans_x)
        attention_output = layers.LayerNormalization()(trans_x + attention_output)
        
        # Feed-forward
        ffn_output = layers.Dense(256, activation="relu")(attention_output)
        ffn_output = layers.Dense(self.transformer_d_model)(ffn_output)
        trans_x = layers.LayerNormalization()(attention_output + ffn_output)
        
        trans_features = layers.GlobalAveragePooling1D()(trans_x)
        
        # Combine features
        combined = layers.Concatenate()([graph_features, trans_features])
        combined = layers.Dense(256, activation='relu')(combined)
        combined = layers.Dropout(0.3)(combined)
        
        # Output layers
        flare_class = layers.Dense(
            self.n_classes, activation='softmax', name='flare_class'
        )(combined)
        
        model = keras.Model(
            inputs=[node_features, adjacency_input],
            outputs=flare_class
        )
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
