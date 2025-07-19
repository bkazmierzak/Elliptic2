import pandas as pd
import torch
import dgl
import os
import pickle
from collections import defaultdict

# --- Configuration ---
DATAPATH = "./dataset/"
OUTPUT_DIR = "./processed_data/"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# TEST_RATIO is the remainder

def preprocess_and_save():
    """
    Loads raw Elliptic2 data, processes it into a scalable DGL graph format,
    and saves the graph and subgraph labels.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load Nodes and Features
    print("Loading background nodes and features...")
    # Load all feature columns
    node_features = pd.read_csv(os.path.join(DATAPATH, "background_nodes.csv"))
    # Create the node ID mapping
    n2id = {cl_id: i for i, cl_id in enumerate(node_features['clId'])}
    
    # Extract feature tensor (excluding the 'clId' column)
    features_tensor = torch.tensor(node_features.iloc[:, 1:].values, dtype=torch.float32)
    num_nodes = len(node_features)
    print(f"Loaded {num_nodes} nodes with {features_tensor.shape[1]} features each.")

    # 2. Load Edges and build graph
    print("Loading background edges...")
    edge_df = pd.read_csv(os.path.join(DATAPATH, "background_edges.csv"))
    src_nodes = torch.tensor([n2id[c1] for c1 in edge_df['clId1']])
    dst_nodes = torch.tensor([n2id[c2] for c2 in edge_df['clId2']])
    
    # Create DGL graph
    g = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
    g.ndata['feat'] = features_tensor
    print("DGL graph constructed.")

    # 3. Process Labeled Subgraphs
    print("Processing labeled subgraphs...")
    cc_df = pd.read_csv(os.path.join(DATAPATH, "connected_components.csv"))
    sub_nodes_df = pd.read_csv(os.path.join(DATAPATH, "nodes.csv"))

    # Group nodes by subgraph ID
    subgraphs = defaultdict(list)
    for _, row in sub_nodes_df.iterrows():
        subgraphs[row['ccId']].append(n2id[row['nodeId']])

    labeled_subgraphs = []
    for _, row in cc_df.iterrows():
        cc_id = row['ccId']
        # Convert label to integer (0 for licit, 1 for suspicious)
        label = 0 if row['ccLabel'] == 'licit' else 1
        nodes = torch.tensor(subgraphs[cc_id], dtype=torch.int32)
        labeled_subgraphs.append({'nodes': nodes, 'label': label})

    # 4. Split and Save Subgraphs
    print("Splitting and saving subgraph data...")
    # This uses a fixed random seed for reproducibility
    torch.manual_seed(42)
    shuffled_subs = [labeled_subgraphs[i] for i in torch.randperm(len(labeled_subgraphs))]
    
    num_train = int(len(shuffled_subs) * TRAIN_RATIO)
    num_val = int(len(shuffled_subs) * VAL_RATIO)
    
    train_set = shuffled_subs[:num_train]
    val_set = shuffled_subs[num_train:num_train + num_val]
    test_set = shuffled_subs[num_train + num_val:]

    # --- Save Outputs ---
    dgl.save_graphs(os.path.join(OUTPUT_DIR, "graph.bin"), [g])
    with open(os.path.join(OUTPUT_DIR, "train_subs.pkl"), "wb") as f:
        pickle.dump(train_set, f)
    with open(os.path.join(OUTPUT_DIR, "val_subs.pkl"), "wb") as f:
        pickle.dump(val_set, f)
    with open(os.path.join(OUTPUT_DIR, "test_subs.pkl"), "wb") as f:
        pickle.dump(test_set, f)
        
    print(f"Preprocessing complete. Data saved to '{OUTPUT_DIR}'.")
    print(f"Training set: {len(train_set)}, Validation set: {len(val_set)}, Test set: {len(test_set)}")

if __name__ == "__main__":
    preprocess_and_save()