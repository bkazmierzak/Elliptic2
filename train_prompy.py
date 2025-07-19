import dgl
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
from models import GNNWithPrompt # A new model class in models.py
from sklearn.metrics import f1_score, roc_auc_score

# --- Configuration ---
DATA_DIR = "./processed_data/"
PRETRAINED_MODEL_PATH = "./models/pretrained_gnn.pth"
EPOCHS = 5
LEARNING_RATE = 0.01 # Higher LR for prompts is common
BATCH_SIZE = 64
NEIGHBOR_SAMPLING_SIZES = [15, 10]

def train_and_evaluate():
    """
    Trains a prompt-based GNN for subgraph classification and evaluates it.
    """
    # 1. Load Graph and Subgraph Sets
    g, _ = dgl.load_graphs(os.path.join(DATA_DIR, "graph.bin"))
    g = g[0]
    
    with open(os.path.join(DATA_DIR, "train_subs.pkl"), "rb") as f:
        train_set = pickle.load(f)

    # We need a custom collate function to handle batches of subgraphs
    def collate_fn(batch):
        subgraph_nodes = [item['nodes'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch])
        return subgraph_nodes, labels
    
    # 2. Model and Optimizer
    model = GNNWithPrompt(
        input_dim=g.ndata['feat'].shape[1],
        hidden_dim=128,
        output_dim=2, # Binary classification
        prompt_dim=16 # Small MLP for the prompt
    )
    
    # Load pretrained weights and freeze the GNN backbone
    model.gnn.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
    for param in model.gnn.parameters():
        param.requires_grad = False
    
    # The optimizer only updates the prompt and classifier weights
    optimizer = optim.Adam(
        list(model.prompt_in.parameters()) + 
        list(model.prompt_out.parameters()) + 
        list(model.classifier.parameters()), 
        lr=LEARNING_RATE
    )
    loss_fn = nn.CrossEntropyLoss()

    # 3. Create Scalable DataLoader
    sampler = dgl_dl.MultiLayerNeighborSampler(NEIGHBOR_SAMPLING_SIZES)
    # We iterate over subgraph indices and fetch them in the collate_fn
    dataloader = dgl_dl.SubgraphDataLoader(
        g, [item['nodes'] for item in train_set], sampler,
        batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)

    print("Starting Prompt-Tuning...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # --- Node Membership Logic ---
            # `output_nodes` contains the IDs of the nodes in the final layer of the computation graph.
            # We need to create the membership mask for all nodes in the *first* block.
            all_block_nodes = blocks[0].srcdata[dgl.NID]
            membership_mask = torch.zeros_like(all_block_nodes, dtype=torch.bool)
            
            # Check which nodes in the block belong to the target subgraphs (`output_nodes`)
            # This is a simplification; a more robust way would track original subgraph nodes.
            is_member = torch.isin(all_block_nodes, output_nodes)
            membership_mask[is_member] = True

            # Get labels for the current batch of subgraphs
            labels = torch.tensor([train_set[idx]['label'] for idx in output_nodes.tolist() if idx < len(train_set)])

            # Forward pass with prompts
            output = model(blocks, blocks[0].srcdata['feat'], membership_mask, output_nodes)

            if output.shape[0] == 0: continue # Skip if no output nodes (can happen in last batch)
            
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {total_loss/len(dataloader):.4f}")

    # --- Evaluation would go here ---
    # You would create a similar loop for the validation/test set,
    # but without the optimizer steps (model.eval() mode).
    print("Training complete.")


if __name__ == "__main__":
    train_and_evaluate()