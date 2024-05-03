import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import pickle
import LSTM
from text_dataset import TextDataset
import multiprocessing

if __name__ == '__main__':
    # Call freeze_support() before any other multiprocessing operations
    multiprocessing.freeze_support()

    # Load the necessary data
    with open('training_data.pkl', 'rb') as f:
        data = pickle.load(f)

    text_paths = data['text_paths']
    seq_length = data['seq_length']
    vocab = data['vocab']
    word_to_idx = data['word_to_idx']
    idx_to_word = data['idx_to_word']

    # Reconstruct the dataset
    dataset = TextDataset(text_paths=text_paths, seq_length=seq_length, vocab=vocab, word_to_idx=word_to_idx, idx_to_word=idx_to_word)

    # Define parameters
    vocab_size = len(dataset.vocab)
    embedding_dim = 256
    hidden_dim = 512
    num_layers = 2
    batch_size = 612
    learning_rate = 0.0005
    num_epochs = 6
    dropout = 0.2

    # Create DataLoader for training
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using: {device}")
    model = LSTM.LSTMLanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers, dropout).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scaler = torch.cuda.amp.GradScaler()

    num_of_batches = len(dataloader)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        counter = 0
        for batch_inputs, batch_targets in dataloader:
            with autocast():  # Use automatic mixed precision
                batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
                optimizer.zero_grad()
                hidden = model.init_hidden(batch_inputs.size(0), device)
                logits, _ = model(batch_inputs, hidden)
                loss = criterion(logits, batch_targets.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
            counter += 1
            print(f"{counter} of {num_of_batches} batches processed in the {epoch + 1} epoch")

        average_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')

    print('Training finished.')

    # Save the trained model
    torch.save(model.state_dict(), 'trained_model.pth')
    print('Model saved.')