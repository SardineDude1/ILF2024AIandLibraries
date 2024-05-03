import torch
import pickle
from LSTM import LSTMLanguageModel
from torchtext.data.utils import get_tokenizer

# Load the necessary data from the pickle file
with open('training_data.pkl', 'rb') as f:
    data = pickle.load(f)

vocab = data['vocab']
vocab_size = len(vocab)
word_to_idx = data['word_to_idx']
idx_to_word = data['idx_to_word']

embedding_dim = 256
hidden_dim = 512
num_layers = 2
batch_size = 612
learning_rate = 0.0005
num_epochs = 6
dropout = 0.2


# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMLanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
model.load_state_dict(torch.load('trained_model.pth', map_location=device))
model.eval()

# Function to generate text
def generate_text(model, seed_text, max_length=20):
    tokenizer = get_tokenizer('basic_english')
    seed_encoded = tokenizer(seed_text.lower())
    seed_indices = [word_to_idx.get(word, 0) for word in seed_encoded]
    
    input_seq = torch.tensor(seed_indices, dtype=torch.long, device=device).unsqueeze(0)
    hidden = model.init_hidden(batch_size=1, device=device)
    
    generated_text = ""
    
    for _ in range(max_length):
        logits, hidden = model(input_seq, hidden)
        last_logits = logits[-1, :]  # Extract logits for the last token
        
        prediction = torch.multinomial(last_logits.exp(), num_samples=1)
        
        word_idx = prediction.item()
        if word_idx == 0:  # Encountered padding token
            break
        
        word = idx_to_word[word_idx]
        generated_text += ' ' + word
        
        input_seq = torch.cat((input_seq, prediction.unsqueeze(1)), dim=1)
    
    return generated_text

# Chat loop
print("Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    elif user_input == "":
        print("Your chat cannot be empty.")
        print("To stop chatting type 'exit'.")
        continue
    generated_response = generate_text(model, user_input)
    print("Bot:", generated_response)
