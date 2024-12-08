import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch

def train_model(model, dataloader, epochs=100, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

def train_variational(model, dataloader, epochs=100, lr=0.01, kl_weight=0.0001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_batch)
            prediction_loss = F.cross_entropy(outputs, y_batch)
            kl_loss = model.kl_divergence()
            
            # Total loss
            loss = prediction_loss + kl_weight * kl_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

def train_flipout_model(model, train_loader, epochs=100, learning_rate=1e-2, kl_weight=1e-4):
    # Set device and move model to the device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            # Move batch data to device
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            nll_loss = F.cross_entropy(logits, y_batch)
            kl_loss = model.kl_divergence()
            loss = nll_loss + kl_weight * kl_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Optional: Print the average loss for the epoch
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

def evaluate_flipout_model(model, x, y, n_samples=50):
    model.eval()
    mean, variance = model.predict_mean_and_variance(x, n_samples)
    predictions = mean.argmax(dim=1)
    accuracy = (predictions == y).float().mean().item()
    return accuracy, mean, variance