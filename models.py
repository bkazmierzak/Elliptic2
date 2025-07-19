import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn

class GNN(nn.Module):
    """
    A standard Graph Neural Network model (e.g., GraphSAGE).
    This is used as the backbone for both pre-training and prompt-tuning.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GNN, self).__init__()
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(dglnn.SAGEConv(input_dim, hidden_dim, 'mean'))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(dglnn.SAGEConv(hidden_dim, hidden_dim, 'mean'))
        # Output layer
        self.layers.append(dglnn.SAGEConv(hidden_dim, output_dim, 'mean'))
        self.activation = nn.ReLU()

    def forward(self, blocks, features):
        h = features
        for i, layer in enumerate(self.layers):
            h_dst = h[:blocks[i].num_dst_nodes()]
            h = layer(blocks[i], (h, h_dst))
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

class GNNWithPrompt(nn.Module):
    """
    Implements the MPrompt methodology. It wraps a GNN backbone, freezes it,
    and applies trainable prompts based on node membership.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, prompt_dim):
        super(GNNWithPrompt, self).__init__()
        
        # The GNN backbone (will be frozen during prompt-tuning)
        self.gnn = GNN(input_dim + prompt_dim, hidden_dim, hidden_dim, num_layers=2)
        
        # Small MLPs for the prompts
        self.prompt_in = nn.Sequential(nn.Linear(input_dim, prompt_dim), nn.ReLU())
        self.prompt_out = nn.Sequential(nn.Linear(input_dim, prompt_dim), nn.ReLU())
        
        # Classifier for the final subgraph representation
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, blocks, features, membership_mask, output_nodes_indices):
        # 1. Apply Prompts
        prompt_features_in = self.prompt_in(features)
        prompt_features_out = self.prompt_out(features)
        
        # Select the correct prompt based on membership
        # Create a tensor of shape (num_nodes, prompt_dim)
        selected_prompts = torch.where(
            membership_mask.unsqueeze(1), 
            prompt_features_in, 
            prompt_features_out
        )
        
        # Concatenate original features with prompt features
        prompted_features = torch.cat([features, selected_prompts], dim=1)
        
        # 2. Forward pass through the (frozen) GNN
        node_embeddings = self.gnn(blocks, prompted_features)
        
        # 3. Pool node embeddings for each subgraph
        # We need a way to map from the output_nodes (which are contiguous in the block)
        # to the original subgraph groups. This requires a more complex dataloader setup.
        # For simplicity, we assume one output node per subgraph for this example.
        subgraph_embeddings = node_embeddings # This needs to be replaced with actual pooling
        
        # 4. Classify
        logits = self.classifier(subgraph_embeddings)
        return logits