import numpy as np
import NeuralNetwork as nn

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

    def choose_action(self, input_state):
        if type(input_state) is list:
            input_state = np.array(input_state)
        if input_state.shape[0] != self.nn_structure.get_number_of_inputs():
            raise ValueError('Input is incorrect size for this Agent')
        return self.network.evaluate(input_state)


if __name__ == "__main__":
    np.random.seed(1)
    nn_struct = nn.NetworkStructure([4,3,2])
    A = Agent(nn_struct)
    x = np.array([4,10,-1,2])
    print(A.choose_action(x))