import numpy as np
from snn.network.snn import SNN

"""
Reward Based Spiking Neural Network model
"""
class RewardSNN(SNN):
  def __init__(self, num_input, hidden_layers, num_output, neuron_class, learning_method, reward_controller):
    super(self.__class__, self).__init__(num_input, hidden_layers, num_output, neuron_class, learning_method)
    self.reward_controller = reward_controller

  def solve(self, input, individual):
    previous_layer = np.array(input)
    for (i, layer) in enumerate(self.layers):
      new_previous_layer = np.array([])
      for neuron in layer:
        neuron.reward = self.reward_controller(individual)
        new_previous_layer = np.append(new_previous_layer, neuron.solve(previous_layer))
      previous_layer = new_previous_layer
    self.adjust_weights()
    return previous_layer
