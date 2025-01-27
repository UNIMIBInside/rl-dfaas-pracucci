import pathlib

import neat
import matplotlib.pyplot as plt
from env.env import TrafficManagementEnv

class ExtendedStatisticsReporter(neat.StatisticsReporter):

    def __init__(self):
        super().__init__()
        self.generation = 0

        # Create logs folder in the root project directory.
        self.logs_folder = pathlib.Path(__file__).parent.parent.parent / 'logs' / 'NEAT'
        self.logs_folder.mkdir(parents=True, exist_ok=True)
        print(f"Logs folder: {str(self.logs_folder)!r}")

    def post_evaluate(self, config, population, species, best_genome):
        super().post_evaluate(config, population, species, best_genome)

        # Stampa le statistiche ogni N generazioni
        if self.generation % 5 == 0:
            print("\n----- Statistics till Generation {} -----".format(self.generation))
            print("Mean fitness: {:.2f}".format(self.get_fitness_mean()[-1]))
            print("Max fitness: {:.2f}".format(self.get_fitness_stdev()[-1]))

    def end_generation(self, config, population, species):
        super().end_generation(config, population, species)

        # Salva grafici ogni N generazioni
        if self.generation % 10 == 0:
            self.plot_statistics()
        self.generation += 1

    def plot_statistics(self):
        fig, ax = plt.subplots()

        # Retrieve the max fitness values for each generation
        max_fitness_values = [genome.fitness for genome in self.most_fit_genomes]
        ax.plot(max_fitness_values, label="Max Fitness")

        # Retrieve the mean fitness values for each generation and plot
        mean_fitness_values = self.get_fitness_mean()
        ax.plot(mean_fitness_values, label="Mean Fitness", linestyle='--')  # Using a dashed line for mean fitness

        ax.set(xlabel='Generation', ylabel='Fitness', title='Fitness over Generations')
        ax.grid()
        ax.legend()

        fig_path = self.logs_folder / f'neat_fitness_plot_gen{self.generation}.png'
        plt.savefig(fig_path)
        plt.close()

def eval_genomes(genomes, config):
    env = TrafficManagementEnv()
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        ob = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = net.activate(ob)
            ob, reward, _, done, _ = env.step(action)
            total_reward += reward
        genome.fitness = total_reward
        if genome.fitness is None:
            print("Warning: Fitness is None for genome_id", genome_id)

def run():
    config_path = pathlib.Path(__file__).with_name('config_optimized.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)

    # Save checkpoints data to "checkpoints" folder on root project directory.
    # Each checkpoint have 'neat-checkpoint-' prefix.
    checkpoint_folder = pathlib.Path(__file__).parent.parent.parent / 'checkpoints'
    print(f'Checkpoint folder: {str(checkpoint_folder)!r}')
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    checkpoint_prefix = checkpoint_folder / 'neat-checkpoint-'

    # Save checkpoint every 50 generations.
    checkpointer = neat.Checkpointer(50, None, checkpoint_prefix)
    pop.add_reporter(checkpointer)

    stats = ExtendedStatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    winner = pop.run(eval_genomes, 100)

    return winner, stats
