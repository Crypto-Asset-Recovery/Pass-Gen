# Pass-Gen: An LSTM RNN approach to generating likely password candidates

This is a project to generate likely password candidates using an LSTM RNN. Currently, the model is trained on a dataset of the top 100 thousand passwords from the [SecList's Common Credentials](https://github.com/danielmiessler/SecLists/tree/master/Passwords/Common-Credentials) dataset. The data is split into training and validation sets (20/80), and the model is trained on the training set. The model is then evaluated on the validation set. The model is then saved (trained_model.pth) and used to generate likely password candidates.

You can use the pregenerated model using the command: `python3 load_model.py --load_num_layers 8` Note: currently you need to load 8 layers or retrain a new model on a different number of layers.

The pretrained model has the following parameters:
- Number of layers: 8
- Hidden size: 256
- Dropout: 0.5
- Learning rate: 0.0035
- Epochs: 5

## Requirements

- Python 3.9+
- PyTorch 2.0+
- argparse
- sklearn
Probably some other stuff, but I can't remember :P