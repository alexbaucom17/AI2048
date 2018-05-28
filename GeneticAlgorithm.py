import numpy as np
import NeuralNetwork as nn
import game2048
import math
import game2048_gui as gui
import tkinter as tk
import time
import pickle
import concurrent.futures

SCALE = 1 #this allows dna values to randomly initialize in range [-1 1]
MOVE_PAUSE_TIME = 0.15 #seconds to pause in between moves during replay


def genome_crossover(genome1, genome2):
    #safety check
    if genome1.get_length() != genome2.get_length():
        raise ValueError('Genomes are not the same length!')
    #initialize
    n = int(np.random.randint(1, genome1.get_length(), 1))
    dna1 = genome1.get_dna()
    dna2 = genome2.get_dna()
    #perform crossover
    tmp = np.copy(dna1[n:])
    dna1[n:] = dna2[n:]
    dna2[n:] = tmp
    #copy to new instances
    structure = genome1.get_structure()
    genome3 = Genome(structure)
    genome4 = Genome(structure)
    genome3.set_dna(dna1)
    genome4.set_dna(dna2)
    return genome3, genome4


class Genome:

    def __init__(self, structure):
        self.structure = structure
        self.gene_sizes = [self.structure.get_number_of_weights_per_layer(i) for i in range(self.structure.get_number_of_layers())]
        self.gene_start_index = np.cumsum(self.gene_sizes) - self.gene_sizes
        self.genome_length = sum(self.gene_sizes)
        self.dna = SCALE * (np.random.uniform(-SCALE, SCALE, self.genome_length ))

    def get_length(self):
        return self.genome_length

    def get_structure(self):
        return self.structure

    def copy(self):
        new_structure = self.structure.copy()
        new_dna = self.get_dna()
        new_genome = Genome(new_structure)
        new_genome.set_dna(new_dna)
        return new_genome

    def set_dna(self,dna):
        if len(dna) != self.genome_length:
            raise ValueError('DNA sequence is of incorrect length!')
        self.dna = dna

    def get_dna(self):
        return np.copy(self.dna)

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

    def mutate(self,rate):
        mutation_dna = np.random.uniform(-SCALE, SCALE, self.genome_length )
        self.dna = np.array([self.dna[i] if np.random.uniform() > rate else mutation_dna[i] for i in range(self.genome_length)])



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
        self.game_random_state = np.random.RandomState(np.random.randint(10000)) #ensures that this random state is produced from a seeded stream
        self.previous_game_random_seed = -1
        if self.game_size % 1 != 0:
            raise ValueError('Input layer must be a square number!')
        else:
            self.game_size = int(self.game_size)

    def get_genome(self):
        return self.genome

    def mutate_genome(self,rate):
        self.genome.mutate(rate)

    def copy(self):
        new_genome = self.genome.copy()
        new_struct = new_genome.get_structure()
        return Agent(new_struct,new_genome)

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


    def play_game(self, gui_root=None):

        self.previous_game_random_seed = self.game_random_state.get_state()
        game = game2048.game2048(self.game_size,random_stream=self.game_random_state)
        game_over = False
        win = False

        self.stuck_counter = 0
        self.old_state = game.get_state(flat=True)

        if gui_root:
            GUI = gui.Board(gui_root,game)
            GUI.pack(side=tk.LEFT, padx=1, pady=1)

        while not game_over:
            flat_state = np.copy(game.get_state(flat=True))
            action = self.choose_action(flat_state)
            game.swipe(action)
            (game_over,win) = self.check_for_game_over(game, flat_state)
            if gui_root:
                GUI.update_tiles()
                gui_root.update()
                time.sleep(MOVE_PAUSE_TIME)

        return game.get_score(),win

    def replay_previous_game(self, gui_root):
        if self.previous_game_random_seed != -1:
            old_seed = self.game_random_state.get_state()
            self.game_random_state.set_state(self.previous_game_random_seed)
            score,win = self.play_game(gui_root)
            tmp = self.game_random_state.get_state()
            self.game_random_state.set_state(old_seed)
            return score,win
        else:
            raise ValueError("A game must be played before attempting to replay it!")



def StochasticSamplingWithoutReplacement(population_scores_in,number_to_keep):
    idx_to_keep = []
    population_scores = np.copy(population_scores_in)
    for i in range(number_to_keep):
        total_score = sum(population_scores)
        selection_point = np.random.uniform(0,total_score,1)
        idx = RouletteWheelSelection(population_scores,selection_point)
        idx_to_keep = np.append(idx_to_keep, idx)
        population_scores[idx] = 0
    return np.array(idx_to_keep)

def StochasticUniversalSampling(population_scores, number_to_keep):
    total_score = sum(population_scores)
    delta = total_score/number_to_keep
    start_point = np.random.uniform(0,delta,1)
    points = np.array([start_point + i*delta for i in range(number_to_keep-1)])
    return RouletteWheelSelection(population_scores,points)

def RouletteWheelSelection(scores,selection_points):
    idx_to_keep = []
    cummulateive_scores = np.cumsum(scores)
    count = 0
    for P in selection_points:
        while cummulateive_scores[count] < P:
            count = count + 1
        idx_to_keep.append(count)
    return np.array(idx_to_keep)


class GeneticLearner:

    def __init__(self, n_agents, layer_sizes, seed=0):
        np.random.seed(seed)
        self.nn_struct = nn.NetworkStructure(layer_sizes)
        self.n_agents = n_agents
        self.agents = [Agent(self.nn_struct) for i in range(self.n_agents)] #randomly initialize all agents
        self.agent_scores = []
        self.win = []

        #learning parameters
        self.survival_rate = 0.5 #what percentage of population to keep
        self.mutation_rate = 0.001 #
        self.crossover_rate = 0.7 #
        self.best_always_survives = True

        #other initialization
        self.n_to_keep = int(self.n_agents * self.survival_rate)
        self.n_to_create = self.n_agents - self.n_to_keep
        #make sure the number of offspring is even to match with 2 parents per offspring pair
        if self.n_to_create % 2 != 0:
            self.n_to_keep = self.n_to_keep + 1
            self.n_to_create = self.n_to_create - 1

        #initialize population scores
        self.population_fitness_test()

    def run_one_generation(self):
        self.select_survivors()
        self.generate_offspring()
        self.mutate()
        self.population_fitness_test()

    def population_fitness_test(self):
        self.agent_scores = [self.agents[i].play_game() for i in range(self.n_agents)]

    def population_fitness_test_multithread(self):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [ executor.submit(self.agents[i].play_game) for i in range(self.n_agents)]
            concurrent.futures.wait(futures)
            self.agent_scores = [f.result() for f in futures]


    def select_survivors(self):
        scores = np.array([self.agent_scores[i][0] for i in range(self.n_agents)])

        if self.best_always_survives:
            idx_best = np.argmax(scores)
            scores[idx_best] = 0
            idx_survived = StochasticSamplingWithoutReplacement(scores,self.n_to_keep-1)
            idx_survived = np.append(idx_survived, idx_best )
        else:
            idx_survived = StochasticSamplingWithoutReplacement(scores, self.n_to_keep)

        idx_survived = tuple(idx_survived)
        self.agents = [ self.agents[i] for i in range(self.n_agents) if i in idx_survived]
        self.agent_scores = [ self.agent_scores[i] for i in range(self.n_agents) if i in idx_survived]

    def select_parents(self):
        cur_num_agents = len(self.agent_scores)
        pairs_of_parents_needed = int(self.n_to_create/2)
        scores = np.array([self.agent_scores[i][0] for i in range(cur_num_agents)])
        return [tuple(StochasticSamplingWithoutReplacement(scores, 2)) for i in range(pairs_of_parents_needed)]

    def generate_offspring(self):
        parent_idx = self.select_parents()
        new_generation = []
        for p1idx,p2idx in parent_idx:
            if np.random.uniform() < self.crossover_rate:
                genome1,genome2 = genome_crossover(self.agents[int(p1idx)].get_genome(),self.agents[int(p2idx)].get_genome())
                new_generation = new_generation + [Agent(self.nn_struct,genome1),Agent(self.nn_struct,genome2)]
            else:
                new_generation = new_generation + [self.agents[int(p1idx)].copy(), self.agents[int(p1idx)].copy()]
        self.agents = self.agents + new_generation

    def mutate(self):
        for i in range(self.n_agents):
            self.agents[i].mutate_genome(self.mutation_rate)

    def get_current_generation_statistics(self):
        scores = self.get_score_array()
        return np.max(scores), np.median(scores), np.min(scores)

    def run_n_generations(self,n):
        for i in range(n):
            self.run_one_generation()
            max, med, min = self.get_current_generation_statistics()
            s = "Generation " + str(i) + " - max: " + str(max) + "  median: " + str(med) + " min: " + str(min)
            print(s)

    def get_score_array(self):
        return np.array([self.agent_scores[i][0] for i in range(self.n_agents)])

    def get_best_agent(self):
        scores = self.get_score_array()
        best_idx = np.argmax(scores)
        return self.agents[best_idx]

    def get_worst_agent(self):
        scores = self.get_score_array()
        worst_idx = np.argmin(scores)
        return self.agents[worst_idx]

    def get_median_agent(self):
        scores = self.get_score_array()
        med_val = np.median(scores)
        med_idx = np.argmax(scores == med_val) #returns only first value where scores == med_val
        return self.agents[med_idx]


def save_model_state(model, fname):
    pickle.dump(model, open(fname, "wb"))

def load_model_state(fname):
    return pickle.load(open(fname, "rb"))


#TODO
#Useful GUI
#consider testing on multiple rounds of the game
#optimization


if __name__ == "__main__":
    #np.random.seed(4)
    #nn_struct = nn.NetworkStructure([16,8,4])
    #A = Agent(nn_struct)
    #A.mutate_genome(0.7)
    #print(A.play_game())

    # fname = 'model_20180122_'
    # n_agents = 1500
    # generations_per_batch = 50
    # n_batches = 100
    # total_generations = generations_per_batch*n_batches
    # print('Preparing to run ' + str(total_generations) +' generations...')
    #
    # G = GeneticLearner(n_agents,[16,8,4],seed=4)
    # #G.run_n_generations(5)
    # #print(G.agent_scores)
    #
    # for i in range(n_batches):
    #     print("Starting batch "+str(i))
    #     G.run_n_generations(generations_per_batch)
    #     gen_num = i*generations_per_batch
    #     save_model_state(G,fname + str(gen_num) + '.p')
    #     print("Model " + str(gen_num) + " Saved")

    fname = 'model_20180122_200'
    G = load_model_state(fname + '.p')
    A = G.get_best_agent()
    root = tk.Tk()
    root.title("2048 Game")
    A.replay_previous_game(root)
    time.sleep(0.5)
    A.play_game(root)
    time.sleep(0.5)
    A.play_game(root)
    time.sleep(0.5)
    A.play_game(root)
    time.sleep(0.5)
    input("Press Enter to end...")

    #print(A.genome.dna)
    #print(np.any(np.abs(A.genome.dna) > 1))
    # dna_vals_greater_than_one = 0
    # for a in G.agents:
    #     if np.any(np.abs(a.genome.dna) > 0.99):
    #         dna_vals_greater_than_one += 1
    #         print(dna_vals_greater_than_one)







    #G.population_fitness_test()
    #print(G.agent_scores)
    #G.select_survivors()
    #print(G.agent_scores)
    #print(G.select_parents())

    #A = np.random.uniform(0,50,20)
    #A[7] = 5
    #print(A)
    #print(StochasticUniversalSampling(A,10))
    #print(StochasticSamplingWithoutReplacement(A, 10))

    #s = nn.NetworkStructure([1,2])
    #g1 = Genome(s)
    #g2 = Genome(s)
    #print(g1.get_dna())
    #print(g2.get_dna())
    #g3,g4 = genome_crossover(g1,g2)
    #print(g3.get_dna())
    #print(g4.get_dna())
