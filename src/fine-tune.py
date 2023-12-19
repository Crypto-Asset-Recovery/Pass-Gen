from src.prepareData import prepareData
import argparse, torch
from src.preprocessData import get_dataloaders

from torch import nn
from src.model import RNN
from src.train import train
from src.generate import generate_password

def load_new_data(existing_vocab):
    # Load your new dataset of 5-10 passwords here
    with open("wordlists/user-ext.txt", "r") as f:
        new_passwords = f.readlines()
    
    # Use the prepareData function to process the new dataset
    train_data, val_data, vocab = prepareData(data=new_passwords, existing_vocab=existing_vocab)

    return train_data, val_data, vocab

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune a model')
    parser.add_argument('--model_path', default='models/250k-long-epoch-5-layers-8-lr-0035.pth', type=str)
    parser.add_argument('--embedding_size', default=128, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--num_layers', default=8, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--seq_len', default=20, type=int)
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--temperature', default=0.5, type=float)
    parser.add_argument('--password_length', default=10, type=int)
    parser.add_argument('--num_passwords', default=20, type=int)
    args = parser.parse_args()
    
    train_data, val_data, existing_vocab = prepareData()
    new_train_loader, new_val_loader = get_dataloaders(train_data, val_data, args.batch_size, args.seq_len) # batch_size, seq_len

    new_output_dim = len(existing_vocab)  # The size of the original vocabulary

    pretrained_model = RNN(len(existing_vocab), args.embedding_size, args.hidden_size, args.num_layers, args.dropout)
    pretrained_model.load_state_dict(torch.load(args.model_path))

    pretrained_model.fc = nn.Linear(pretrained_model.hidden_size, new_output_dim)

    # Set requires_grad to False for all layers except the output layer
    for name, param in pretrained_model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False

    lr_embedding = 0.0001
    lr_rnn = 0.001
    lr_fc = 0.005
    lrs = [lr_embedding, lr_rnn, lr_fc]

    trained_model = train(
        pretrained_model,
        new_train_loader,
        new_val_loader,
        args.num_epochs,
        #args.lr,
        lrs,
        args.gradient_accumulation_steps
    )  # Train the model

    # Save the fine-tuned model
    torch.save(trained_model.state_dict(), f"models/fine-tuned-{args.num_epochs}-epochs-{args.gradient_accumulation_steps}-gradient-accumulation.pth")

