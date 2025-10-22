import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import time

def load_data():
    """Loads the Cora dataset."""
    print("Loading Cora dataset...")
    # PyG will automatically download and process the dataset
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]  # Get the one and only graph in the dataset
    
    # Print some basic statistics about the graph
    print(f'Dataset: {dataset.name}')
    print('-----------------------')
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of node features: {dataset.num_node_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print(f'Number of edges: {data.num_edges}')
    
    return data, dataset.num_node_features, dataset.num_classes

class GCN(torch.nn.Module):
    """
    Graph Convolutional Network model
    Two GCN layers
    """
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(42)  # for reproducibility
        
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # x: Node features [num_nodes, num_features]
        # edge_index: Graph connectivity [2, num_edges]

        # 1. First GCN layer + ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # 2. Add dropout for regularization
        x = F.dropout(x, p=0.5, training=self.training)
        
        # 3. Second GCN layer
        x = self.conv2(x, edge_index)
        
        # 4. Output logits
        return x

def train(model, data, optimizer, criterion):
    """Training function for one epoch."""
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Clear old gradients
    
    # Perform a single forward pass
    out = model(data.x, data.edge_index)
    
    # Calculate loss ONLY on the training nodes
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    
    # Perform backpropagation
    loss.backward()
    optimizer.step()
    
    return loss.item()

def test(model, data):
    """Evaluation function."""
    model.eval()  # Set the model to evaluation mode
    out = model(data.x, data.edge_index)
    
    # Get the predictions by finding the class with the highest logit
    pred = out.argmax(dim=1)
    
    # Check accuracy on the test nodes
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    
    # Check accuracy on the validation nodes
    val_correct = pred[data.val_mask] == data.y[data.val_mask]
    val_acc = int(val_correct.sum()) / int(data.val_mask.sum())
    
    return test_acc, val_acc

def main():
    """Main execution function."""
    
    # --- 1. Load Data ---
    data, num_features, num_classes = load_data()

    # --- 2. Setup Model ---
    model = GCN(num_features=num_features,
                hidden_channels=16,
                num_classes=num_classes)
    
    print("\nModel Architecture:")
    print(model)

    # --- 3. Setup Optimizer & Loss ---
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # --- 4. Run Training & Evaluation ---
    num_epochs = 200
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        loss = train(model, data, optimizer, criterion)
        
        if epoch % 10 == 0:
            test_acc, val_acc = test(model, data)
            print(f'Epoch: {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}')

    end_time = time.time()
    print("\nTraining complete!")
    print(f"Total training time: {end_time - start_time:.2f} seconds")
    
    # --- 5. Final Results ---
    final_test_acc, final_val_acc = test(model, data)
    print(f'Final Validation Accuracy: {final_val_acc:.4f}')
    print(f'Final Test Accuracy: {final_test_acc:.4f}')

if __name__ == "__main__":
    main()
