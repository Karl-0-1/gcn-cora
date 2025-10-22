# GCN for Node Classification on the Cora Dataset

This project is a simple implementation of a Graph Convolutional Network (GCN) to perform node classification on the Cora citation network dataset. The model learns to predict the academic topic of a paper based on its content and its citation links to other papers.

This code uses the [PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/en/latest/) library.

## Project Goal

The goal is to demonstrate the effectiveness of GNNs. By using both the node features (a paper's bag-of-words) and the graph structure (its citations), the GCN can achieve significantly higher accuracy than a model that only looks at the paper's content alone.

## 📊 Dataset: Cora

* **Nodes:** 2,708 scientific publications.
* **Edges:** 5,429 citation links between papers.
* **Node Features:** A 1,433-dimension binary vector for each paper, indicating the presence (1) or absence (0) of specific words from a dictionary.
* **Labels:** Each paper is categorized into one of 7 topics (e.g., "Neural Networks", "Theory").

## 🧠 Model: Graph Convolutional Network (GCN)

The model is a simple 2-layer GCN:
1.  **Layer 1:** A `GCNConv` layer that takes the 1,433 node features and aggregates neighborhood information to create 16-dimensional "hidden" embeddings for each node. This is followed by a `ReLU` activation.
2.  **Layer 2:** A final `GCNConv` layer that takes the 16-dimensional embeddings and aggregates neighborhood information one more time to produce the final 7-dimensional logit scores for each of the 7 classes.

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Karl-0-1/gcn-cora.git](https://github.com/Karl-0-1/gcn-cora.git)
    cd gcn-cora-project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the requirements:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: PyTorch and PyG installation can sometimes be complex. If the above fails, please follow the official installation guides for [PyTorch](https://pytorch.org/get-started/locally/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).*

4.  **Run the training script:**
    ```bash
    python train_gcn.py
    ```

## 📈 Results

You should see the model train for 200 epochs. The output will show the training loss and validation/test accuracy every 10 epochs.

**Expected Output:**
```
Loading Cora dataset...
Dataset: Cora
-----------------------
Number of nodes: 2708
Number of node features: 1433
Number of classes: 7
Number of edges: 10556

Model Architecture:
GCN(
  (conv1): GCNConv(1433, 16)
  (conv2): GCNConv(16, 7)
)

Starting training...
Epoch: 010 | Loss: 1.0503 | Val Acc: 0.7040 | Test Acc: 0.7220
Epoch: 020 | Loss: 0.3275 | Val Acc: 0.7620 | Test Acc: 0.7810
Epoch: 030 | Loss: 0.1499 | Val Acc: 0.7740 | Test Acc: 0.7930
...
Epoch: 200 | Loss: 0.0718 | Val Acc: 0.7840 | Test Acc: 0.8060

Training complete!
Total training time: X.XX seconds
Final Validation Accuracy: 0.7840
Final Test Accuracy: 0.8060
```
The final test accuracy should be around **80-81%**.
