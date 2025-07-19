import dgl
import torch
import torch.nn as nn
import torch.optim as optim
import dgl.function as fn
import dgl.dataloading as dgl_dl
import os
from models import GNN # We will define this model in models.py

# --- Configuration ---
DATA_DIR = "./processed_data/"
MODEL_SAVE_PATH = "./models/pretrained_gnn.pth"
EPOCHS = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NEIGHBOR_SAMPLING_SIZES = [15, 10] # For a 2-layer GNN

def srp_loss(positive_sim, negative_sim):
    """
    A simple contrastive loss for SRP.
    Tries to make positive similarity high (close to 1) and negative low (close to 0).
    """
    return -(torch.log(positive_sim).mean() + torch.log(1 - negative_sim).mean())

def train_srp():
    """
    Trains a GNN using the Subgraph Reconstructive Pretraining objective.
    """
    # 1. Load Graph
    g, _ = dgl.load_graphs(os.path.join(DATA_DIR, "graph.bin"))
    g = g[0]
    
    # 2. Model and Optimizer
    # The GNN model is defined in a separate file for modularity
    model = GNN(input_dim=g.ndata['feat'].shape[1], hidden_dim=128, output_dim=128, num_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Create a custom dataset for SRP
    # For simplicity, we randomly sample starting nodes for our subgraphs
    all_nodes = g.nodes()
    srp_dataset = torch.randperm(all_nodes.shape[0])[:10000] # Use 10k random nodes as seeds for pre-training

    sampler = dgl_dl.MultiLayerNeighborSampler(NEIGHBOR_SAMPLING_SIZES)
    # This dataloader will generate batches of 'blocks' for scalable training
    dataloader = dgl_dl.NodeDataLoader(
        g, srp_dataset, sampler,
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)

    print("Starting SRP Pre-training...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Get embeddings for the nodes in the current computational graph
            node_embeddings = model(blocks, blocks[0].srcdata['feat'])

            # --- SRP Logic ---
            # This is a simplified SRP. A full version would reconstruct k-vertex subgraphs.
            # Here, we contrast a node with its neighbors vs. random nodes in the batch.
            
            # Simple contrast: Positive pairs are connected nodes, negative are not.
            # This serves as a proxy for the more complex SRP.
            sub_g = blocks[-1] # The final block contains the target nodes and their direct neighbors
            pos_src, pos_dst = sub_g.edges()
            
            # Use dot product for similarity
            pos_sim = torch.sigmoid((node_embeddings[pos_src] * node_embeddings[pos_dst]).sum(1))
            
            # Negative sampling: contrast with random nodes in the same batch
            neg_src = pos_src
            neg_dst = pos_dst[torch.randperm(len(pos_dst))]
            neg_sim = torch.sigmoid((node_embeddings[neg_src] * node_embeddings[neg_dst]).sum(1))

            loss = srp_loss(pos_sim, neg_sim)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if i % 50 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Batch {i}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f}")

    # --- Save the Model ---
    if not os.path.exists("./models/"):
        os.makedirs("./models/")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Pre-trained model saved to '{MODEL_SAVE_PATH}'")

if __name__ == "__main__":
    train_srp()