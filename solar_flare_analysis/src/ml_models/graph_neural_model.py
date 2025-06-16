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
        Forward pass through graph attention layer
        """
        batch_size = tf.shape(inputs)[0]
        num_nodes = tf.shape(inputs)[1]
        
        # Linear transformation
        h = self.W(inputs)  # [batch, num_nodes, units * num_heads]
        h = tf.reshape(h, [batch_size, num_nodes, self.num_heads, self.units])
        
        # Attention mechanism
        h_i = tf.expand_dims(h, axis=2)  # [batch, num_nodes, 1, num_heads, units]
        h_j = tf.expand_dims(h, axis=1)  # [batch, 1, num_nodes, num_heads, units]
        
        # Concatenate for attention computation
        concat = tf.concat([
            tf.tile(h_i, [1, 1, num_nodes, 1, 1]),
            tf.tile(h_j, [1, num_nodes, 1, 1, 1])
        ], axis=-1)  # [batch, num_nodes, num_nodes, num_heads, 2*units]
        
        # Compute attention scores
        e = self.a(concat)  # [batch, num_nodes, num_nodes, num_heads, 2*units]
        e = tf.reduce_sum(e, axis=-1)  # [batch, num_nodes, num_nodes, num_heads]
        e = self.leaky_relu(e)
        
        # Apply adjacency mask
        adjacency_mask = tf.expand_dims(adjacency_matrix, axis=-1)
        e = tf.where(adjacency_mask > 0, e, -1e9)
        
        # Softmax attention weights
        alpha = tf.nn.softmax(e, axis=2)
        alpha = self.dropout(alpha, training=training)
        
        # Apply attention to node features
        h_prime = tf.einsum('bijk,bjkl->bikl', alpha, h)
        h_prime = tf.reduce_mean(h_prime, axis=-2)  # Average over heads
        
        return h_prime


class GraphNeuralFlareModel:
    """
    Graph Neural Network for modeling complex solar flare relationships
    """
    
    def __init__(self,
                 sequence_length: int = 128,
                 n_features: int = 2,
                 n_classes: int = 6,
                 hidden_units: int = 64,
                 num_gat_layers: int = 3,
                 num_heads: int = 8,
                 k_neighbors: int = 5):
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
            Number of Graph Attention layers
        num_heads : int
            Number of attention heads
        k_neighbors : int
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
        Construct k-NN graph from input data
        """
        batch_size = X.shape[0]
        adjacency_matrices = []
        
        for i in range(batch_size):
            # Flatten sequence for distance computation
            sample = X[i].reshape(self.sequence_length, -1)
            
            # Compute k-NN graph
            adj_matrix = kneighbors_graph(
                sample, n_neighbors=self.k_neighbors,
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
        
        # Initial node embedding
        x = layers.Dense(self.hidden_units, activation='relu')(node_features)
        
        # Graph Attention layers
        for i in range(self.num_gat_layers):
            x = GraphAttentionLayer(
                units=self.hidden_units,
                num_heads=self.num_heads,
                dropout_rate=0.1,
                name=f'gat_layer_{i}'
            )(x, adjacency_input)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        
        # Global graph pooling
        graph_embedding = layers.GlobalAveragePooling1D()(x)
        
        # Final classification layers
        x = layers.Dense(256, activation='relu')(graph_embedding)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
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
        Visualize the constructed graph for a sample
        """
        adjacency = self.construct_graph(X_sample[sample_idx:sample_idx+1])[0]
        
        # Create networkx graph
        G = nx.from_numpy_array(adjacency)
        
        # Plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes and edges
        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                              node_size=100, alpha=0.8)
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5)
        
        plt.title(f'Graph Structure for Sample {sample_idx}')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
