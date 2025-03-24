# Neural Chess Model

A deep learning implementation for chess position understanding based on Leela Chess Zero architecture.

## Project Overview

This project implements a ResNet-based neural network architecture for chess position evaluation, inspired by the Leela Chess Zero (LC0) approach. The model learns to understand chess positions by training on a large dataset of chess games, developing the ability to:

1. Evaluate positions with a policy head (move prediction)
2. Assess game outcomes with a value head (win/loss/draw prediction)

The architecture utilizes modern deep learning techniques including mixed-precision training, residual blocks with squeeze-and-excitation layers, and auxiliary heads for faster training convergence.

**Note:** This repository focuses on the core neural network training components. The embedding work for strategic similarity analysis will be implemented in a separate repository.

## Architecture

The model uses a ChessResNet architecture with the following key components:

- **Input:** 17-plane representation of chess positions (pieces, player turn, castling rights, etc.)
- **Backbone:** Configurable ResNet with residual blocks and SE attention
- **Policy Head:** Predicts move probabilities (73x8x8 output for all possible moves)
- **Value Head:** Predicts game outcome probabilities (win/draw/loss)
- **Auxiliary Heads:** Additional policy and value heads at an intermediate layer for faster training

## Repository Structure

- `model.py` - Neural network architecture implementation
- `training.py` - Training pipeline with optimized data loading and validation
- `pgn_interpreter.py` - PGN parsing and board representation
- `data_loading.py` - Efficient data processing with parallel execution

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA-capable GPU recommended for training

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neural-chess-model.git
cd neural-chess-model

# Install dependencies
pip install -r requirements.txt
```

### Training

To train the model:

```bash
python training.py --data_dir path/to/processed-pgns --save_dir checkpoints --batch_size 512
```

Additional training parameters:
- `--filters`: Number of filters in the model (default: 256)
- `--blocks`: Number of blocks in the model (default: 12)
- `--initial_lr`: Initial learning rate (default: 0.004)
- `--iterations`: Total iterations to train for
- `--valid_every`: Validate every N iterations
- `--auto_resume`: Automatically resume from latest checkpoint

## Data Preparation

The model trains on chess positions from PGN files that have been preprocessed into numpy arrays. The processing pipeline:

1. Parses PGN files using `pgn_interpreter.py`
2. Converts positions into bitboard representations
3. Processes move history into policy targets
4. Saves as compressed numpy files for efficient loading

## Future Work

- Strategic position embedding extraction (to be implemented in a separate repository)
- Weakness detection and targeted training position generation
- Integration with chess.com and lichess.org for game analysis

## License

[MIT License](LICENSE)

## Acknowledgments

- This project draws inspiration from the [Leela Chess Zero](https://github.com/LeelaChessZero/lc0) project
- Training data derived from the Leela self-play games
