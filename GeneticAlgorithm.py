import numpy as np
import NeuralNetwork as nn
import game2048
import math
import game2048_gui as gui
import tkinter as tk
import time

SCALE = 2 #this allows dna values to randomly initialize in range [-1 1]

class Genome:

    def __init__(self, structure):
        self.structure = structure
        self.gene_sizes = [self.structure.get_number_of_weights_per_layer(i) for i in range(self.structure.get_number_of_layers())]
        self.gene_start_index = np.cumsum(self.gene_sizes) - self.gene_sizes
        self.genome_length = sum(self.gene_sizes)
        self.dna = SCALE * (np.random.rand(self.genome_length ) - 0.5)

    def get_flat_genome(self):
        return self.dna

    def get_structured_genome(self):
        weights = []
        bias = []
        for i in range(self.structure.get_number_of_layers()):

            #get sizes
            n_rows = self.structure.get_number_of_neurons_per_layer(i)
            n_cols = self.structure.get_number_of_inputs_per_layer(i)

            #compute indices
            gene_end_index = self.gene_start_index[i] + self.gene_sizes[i]
            weight_end_index = gene_end_index - n_rows # since n_rows is length of bias vector

            #extract flat weights and biases
            weights_flat = self.dna[self.gene_start_index[i]:weight_end_index]
            bias_flat = self.dna[weight_end_index:gene_end_index]

            #append to list
            weights.append(weights_flat.reshape(n_rows,n_cols))
            bias.append(bias_flat)

        return weights, bias

    def generate_network(self):
        (weights, bias) = self.get_structured_genome()
        return nn.Network(weights, bias)


class Agent:

    def __init__(self, nn_structure, genome=None):
        if not type(nn_structure) is nn.NetworkStructure:
            raise ValueError('Input structure must be of NeuralNetwork.NetworkStructure type')

        self.nn_structure = nn_structure

        if genome and type(genome) is Genome:
            self.genome = genome
        else:
            self.genome = Genome(self.nn_structure)

        self.network = self.genome.generate_network()
        self.game_size = math.sqrt(self.nn_structure.get_number_of_inputs())
        if self.game_size % 1 != 0:
            raise ValueError('Input layer must be a square number!')
        else:
            self.game_size = int(self.game_size)


    def evaluate_network(self, input_state):
        if type(input_state) is list:
            input_state = np.array(input_state)
        if input_state.shape[0] != self.nn_structure.get_number_of_inputs():
            raise ValueError('Input is incorrect size for this Agent')
        return self.network.evaluate(input_state)

    def choose_action(self, input_state):
        weighted_actions = self.evaluate_network(input_state)
        max_idx = np.argmax(weighted_actions)
        if max_idx == 0:
            action = 'left'
        elif max_idx == 1:
            action = 'right'
        elif max_idx == 2:
            action = 'up'
        elif max_idx == 3:
            action = 'down'
        else:
            raise ValueError('Invalid action')

        return action

    def check_for_agent_stuck(self, state):
        if np.all(state == self.old_state):
            self.stuck_counter = self.stuck_counter + 1
        else:
            self.stuck_counter = 0
            self.old_state = state

        if self.stuck_counter > 2:
            return True
        else:
            return False

    def check_for_game_over(self, game, flat_state):
        end_state = game.check_for_game_over()
        if end_state == 'WIN':
            return True,True
        elif end_state == 'LOSE':
            return True,False
        elif self.check_for_agent_stuck(flat_state):
            return True,False
        else:
            return False,False

    def play_game(self):
        game = game2048.game2048(self.game_size)

        game_over = False
        self.stuck_counter = 0
        self.old_state = game.get_state(flat=True)
        win = False

        show_gui = True

        if show_gui:
            root = tk.Tk()
            root.title("2048 Game")
            GUI = gui.Board(root,game)
            GUI.pack(side=tk.LEFT, padx=1, pady=1)

        while not game_over:
            flat_state = np.copy(game.get_state(flat=True))
            action = self.choose_action(flat_state)
            game.swipe(action)
            (game_over,win) = self.check_for_game_over(game, flat_state)
            if show_gui:
                GUI.update_tiles()
                root.update()
                time.sleep(0.1)

        if show_gui:
            root.destroy()
        return game.get_score(),win


class GeneticLearner:

    def __init__(self, n_agents, layer_sizes, seed=0):
        np.random.seed(seed)
        self.nn_struct = nn.NetworkStructure(layer_sizes)
        self.n_agents = n_agents
        self.agents = [Agent(self.nn_struct) for i in range(self.n_agents)] #randomly initialize all agents
        self.agent_scores = []


    def test_generation(self):
        self.agent_scores = [self.agents[i].play_game() for i in range(self.n_agents)]



if __name__ == "__main__":
    # np.random.seed(4)
    # nn_struct = nn.NetworkStructure([16,8,4])
    # A = Agent(nn_struct)
    # print(A.play_game())

    G = GeneticLearner(10,[16,8,4],seed=4)
    G.test_generation()
    print(G.agent_scores)
