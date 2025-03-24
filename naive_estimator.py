'''
Estimate base loss for a stupid model
Plan: Scan five compressed npz files, figure out the ratio of wins, draws, and losses
Then figure out the binary cross-entropy of just predicting that ratio regardless of the input
'''

import numpy as np
import os

def load_data(filename):
    with np.load(filename) as data:
        values = data['results']

    wins = np.sum(values == 1)
    draws = np.sum(values == 0)
    losses = np.sum(values == -1)

    return wins, draws, losses

def main():
    data_dir = 'data/processed-pgns'
    filenames = os.listdir(data_dir)
    wins = 0
    draws = 0
    losses = 0
    for filename in filenames[:20]:
        print("Processing", filename)
        w, d, l = load_data(os.path.join(data_dir, filename))
        wins += w
        draws += d
        losses += l

    total = wins + draws + losses
    win_ratio = wins / total
    draw_ratio = draws / total
    loss_ratio = losses / total

    print(f'Win ratio: {win_ratio}')
    print(f'Draw ratio: {draw_ratio}')
    print(f'Loss ratio: {loss_ratio}')

    bce_naive = -win_ratio * np.log(win_ratio) - draw_ratio * np.log(draw_ratio) - loss_ratio * np.log(loss_ratio)
    print(f'Base loss: {bce_naive}')

if __name__ == '__main__':
    main()