# Scalable GNN Training for Elliptic2 Dataset

This project provides a complete, scalable pipeline for training a Graph Neural Network (GNN) on the Elliptic2 dataset. The approach is designed to handle the massive scale of the graph by using modern techniques, including neighborhood sampling and prompt-based fine-tuning, inspired by the **GLASS** and **MPrompt** research papers.

The pipeline is broken down into three main phases:

1.  **Scalable Data Preprocessing**: Converts the raw dataset into an efficient binary graph format with all node features included.
2.  **Self-Supervised Pre-training**: Trains a GNN backbone to learn powerful structural representations of the graph using a self-supervised objective, without using any labels.
3.  **Prompt-Tuning for Classification**: Adapts the powerful pre-trained GNN to the specific task of money laundering detection by training only small, efficient "prompt" modules.

## Directory Structure

For this pipeline to work correctly, please organize your files as follows:

```
.
├── dataset/
│   ├── background_nodes.csv
│   ├── background_edges.csv
│   ├── connected_components.csv
│   └── nodes.csv
├── models/
│   └── (This directory will be created to store model weights)
├── processed_data/
│   └── (This directory will be created to store the processed graph)
├── environment.yml
├── models.py
├── preprocess_scalable.py
├── pretrain_srp.py
└── train_prompt.py
```

## Setup Instructions

These instructions will guide you through setting up a Conda environment with all the necessary dependencies to run the code.

1.  **Clone or Download the Project**:
    Ensure all the Python scripts (`preprocess_scalable.py`, `pretrain_srp.py`, `train_prompt.py`, `models.py`) and the `environment.yml` file are in your project's root directory.

2.  **Create the Conda Environment**:
    Open your terminal, navigate to the project's root directory, and run the following command. This will create a new Conda environment named `elliptic2-gnn` with all the required packages.

    ```bash
    conda env create -f environment.yml
    ```

3.  **Activate the Environment**:
    Once the environment is created, you must activate it before running any of the scripts.

    ```bash
    conda activate elliptic2-gnn
    ```

## How to Run the Test

Follow these steps in order to execute the full training pipeline.

### Step 1: Preprocess the Data

First, run the preprocessing script. This will convert the raw `.csv` files from the `./dataset/` directory into a scalable DGL graph format and save it in the `./processed_data/` directory.

```bash
python preprocess_scalable.py
```

**Expected Output**: The script will print progress updates and confirm when the `graph.bin` and subgraph `.pkl` files have been saved successfully.

### Step 2: Pre-train the GNN Backbone

Next, run the self-supervised pre-training script. This will train the GNN model using the Subgraph Reconstructive Pretraining (SRP) objective and save the weights of the trained backbone to `./models/pretrained_gnn.pth`.

```bash
python pretrain_srp.py
```

**Expected Output**: You will see training progress for each epoch, including the batch number and the current contrastive loss. The script will confirm when the pre-trained model has been saved.

### Step 3: Fine-Tune with Prompts

Finally, run the main training script. This script loads the frozen, pre-trained GNN from Step 2 and fine-tunes the small prompt modules on the labeled subgraph classification task.

```bash
python train_prompt.py
```

**Expected Output**: The script will print the average loss for each epoch of prompt-tuning. After completion, the final model is ready for inference and evaluation.