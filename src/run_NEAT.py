import pickle
import pathlib

from NEAT.neat import run

if __name__ == "__main__":
    winner, stats = run()
    print('\nWinner')
    print(winner)

    # Save the winner genome to the root project directory.
    winner_path = pathlib.Path(__file__).parent.parent / 'neat-winner-genome.pkl'
    with winner_path.open(mode='wb') as file:
        print(f'Saving winner Pickle dump to {str(winner_path)!r}')
        pickle.dump(winner, file)
