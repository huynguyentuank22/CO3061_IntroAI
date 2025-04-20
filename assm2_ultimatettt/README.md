# Ultimate Tic Tac Toe

An implementation of Ultimate Tic Tac Toe with various AI opponents and gameplay modes.

## Requirements

- Python 3.7+
- pygame
- numpy
- torch (for neural network model)
- pandas (for data handling)
- tqdm (for progress bars in simulations)

You can install the required packages using:

```bash
pip install pygame numpy torch pandas tqdm
```

## How to Run

To start the game, run the main.py file:

```bash
python main.py
```

This will open the game menu where you can select different game modes.

## Directory Structure

```
ultimatetictactoe/
├── main.py             # Main entry point for the game
├── game.py             # Game logic and state management
├── board.py            # Board representation and rules
├── agents/             # AI player implementations
│   ├── __init__.py
│   ├── random_agent.py     # Random move selection agent
│   ├── minimax_agent.py    # Minimax algorithm with various difficulty levels
│   ├── mcts_agent.py       # Monte Carlo Tree Search agent
│   └── model_agent.py      # Neural network based agent
├── model.pt            # Pre-trained neural network model
├── train.csv           # Generated training data (created when generating training data)
├── neural/             # Neural network implementation
│   ├── __init__.py
│   ├── model.py        # Neural network architecture
│   └── train.py        # Training script for neural networks
└── README.md           # This file
```

## Game Modes

### Human vs AI
Play against an AI opponent of your chosen difficulty:
- Random: Makes random legal moves
- Easy: Uses minimax with limited depth
- Medium: Uses minimax with moderate depth
- Hard: Uses minimax with greater depth and better evaluation
- MCTS: Uses Monte Carlo Tree Search for decision making
- Model: Uses a pre-trained neural network model

### Agent vs Agent
Watch two AI agents play against each other. Useful for comparing different strategies.

### Batch Simulation
Run multiple games between agents to gather statistics on their performance. Results are displayed after all games are completed.

### Generate Training Data
Create training examples for the neural network by simulating games. The data is saved to train.csv and can be used to train or improve the model.
