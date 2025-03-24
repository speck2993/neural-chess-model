import torch
import numpy as np
import re
import os
import argparse
import glob
from model import ChessResNet
from pgn_interpreter import FastPGNBoard, process_move

def get_best_checkpoint(checkpoint_dir):
    """Find the best model checkpoint in the checkpoints directory."""
    best_pattern = os.path.join(checkpoint_dir, "model_best_*.pt")
    checkpoint_files = glob.glob(best_pattern)
    
    if not checkpoint_files:
        # If no "best" checkpoint exists, find the most recent one
        pattern = os.path.join(checkpoint_dir, "model_*.pt")
        checkpoint_files = glob.glob(pattern)
        
    if not checkpoint_files:
        return None
        
    # Sort by modification time (most recent first)
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    
    return checkpoint_files[0]

def parse_pgn_moves(pgn_file):
    """Extract moves from a PGN file."""
    with open(pgn_file, 'r') as file:
        pgn_content = file.read()
        
    moves = []
    move_pattern = re.findall(r'\b(\d+\.)\s*([^\s.]+)(?:\s+([^\s.]+))?', pgn_content)
    
    for _, white_move, black_move in move_pattern:
        if white_move:
            moves.append(white_move)
        if black_move and not (black_move[0] == "0" or black_move[0] == "1"):
            moves.append(black_move)
            
    return moves

def main():
    """Main function to process a PGN file with the neural chess model."""
    parser = argparse.ArgumentParser(description='Process PGN with neural chess model')
    parser.add_argument('--pgn', required=True, help='Path to PGN file')
    parser.add_argument('--checkpoints', default='checkpoints', help='Path to checkpoints directory')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--no_mixed_precision', action='store_true', help='Disable mixed precision')
    
    args = parser.parse_args()
    
    # Use CPU if CUDA is not available
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    
    # Find best checkpoint
    checkpoint_path = get_best_checkpoint(args.checkpoints)
    if not checkpoint_path:
        print(f"No checkpoint found in {args.checkpoints}")
        return
    
    print(f"Loading model from {checkpoint_path}")
    
    # Load model
    model = ChessResNet()
    model.load(checkpoint_path, device)
    
    # Parse PGN file
    try:
        moves = parse_pgn_moves(args.pgn)
    except Exception as e:
        print(f"Error parsing PGN file: {str(e)}")
        return
    
    # Initialize board
    board = FastPGNBoard()
    player = -1  # Start with white
    
    # Process initial position
    print("Initial Position:")
    board.display()
    
    # Get predictions for initial position
    bitboard = board.board.copy()
    policy_np, value_np = model.predict_position(bitboard, device, not args.no_mixed_precision)
    
    # Display predictions
    print(f"Value: P(white wins)={value_np[0]:.4f}, P(draw)={value_np[1]:.4f}, P(black wins)={value_np[2]:.4f}")
    
    # Find highest probability move
    policy_tensor = policy_np.reshape(73, 8, 8)
    p, r, f = np.unravel_index(policy_tensor.argmax(), policy_tensor.shape)
    print(f"Policy: ({p},{r},{f}) - Highest probability move\n")
    
    # Process each move
    for i, move in enumerate(moves):
        print(f"Move {i+1}: {move}")
        
        try:
            # Process the move
            _, _ = process_move(board, move, player, verbose=False)
            
            # Switch player
            player = -player
            
            # Display board
            board.display()
            
            # Get predictions
            bitboard = board.board.copy()
            policy_np, value_np = model.predict_position(bitboard, device, not args.no_mixed_precision)
            
            # Display predictions
            print(f"Value: P(white wins)={value_np[0]:.4f}, P(draw)={value_np[1]:.4f}, P(black wins)={value_np[2]:.4f}")
            
            # Find highest probability move
            policy_tensor = policy_np.reshape(73, 8, 8)
            p, r, f = np.unravel_index(policy_tensor.argmax(), policy_tensor.shape)
            print(f"Policy: ({p},{r},{f}) - Highest probability move\n")
        except Exception as e:
            print(f"Error processing move {move}: {str(e)}")
            break

if __name__ == "__main__":
    main()