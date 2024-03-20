from torch import nn


# define any number of nn.Modules (or use your current ones)
linear_encoder = nn.Sequential(
    nn.Linear(1000, 500), 
    nn.ReLU(), 
    nn.Linear(500, 250), 
    nn.ReLU(),
    nn.Linear(250, 100), 
    nn.ReLU(),
    nn.Linear(100, 50), 
    nn.ReLU(),
    nn.Linear(50, 20), 
    nn.ReLU(),
    nn.Linear(20, 10),
    )
linear_decoder = nn.Sequential(
    nn.Linear(10, 20), 
    nn.ReLU(),
    nn.Linear(20, 50), 
    nn.ReLU(),
    nn.Linear(50, 100), 
    nn.ReLU(),
    nn.Linear(100, 250), 
    nn.ReLU(),
    nn.Linear(250, 500), 
    nn.ReLU(),
    nn.Linear(500, 1000), 
    nn.Tanh(),
    )