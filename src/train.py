import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import device

def train(model, train_loader, val_loader, num_epochs, lr, gradient_accumulation_steps):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if type(lr) == list:
        # Define the optimizer with different learning rates for each layer group
        optimizer = optim.AdamW([
            {'params': model.embedding.parameters(), 'lr': lr[0]},
            {'params': model.rnn.parameters(), 'lr': lr[1]},
            {'params': model.fc.parameters(), 'lr': lr[2]},
        ])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # Use GPU if available
    model.to(device)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True)

    # Train model
    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0

        # Train on training set
        model.train()
        hidden = model.init_hidden(train_loader.batch_size)
        for i, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output, hidden = model(inputs, hidden)

            # Detach hidden state
            hidden = tuple([h.detach() for h in hidden])

            # Compute loss and backward pass
            loss = criterion(output.view(-1, output.shape[-1]), targets.view(-1))
            loss.backward()

            # Accumulate gradients
            if (i + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            hidden = model.init_hidden(val_loader.batch_size)
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                output, hidden = model(inputs, hidden)

                # Detach hidden state
                hidden = tuple([h.detach() for h in hidden])

                # Compute loss
                loss = criterion(output.view(-1, output.shape[-1]), targets.view(-1))

                val_loss += loss.item()

        # Print progress
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Print learning rates
        for i, group in enumerate(optimizer.param_groups):
            print(f"Learning rate for parameter group {i}: {group['lr']:.6f}")

        scheduler.step(val_loss)

    return model

