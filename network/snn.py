import numpy as np

"""
Spiking Neural Network model
"""
class SNN:
  def __init__(self, num_input, hidden_layers, num_output, neuron_class, learning_method):
    self.layers = []
    self.neuronClass = neuron_class
    self.learning = learning_method
    self.setup(num_input, hidden_layers, num_output)

  def setup(self, num_input, hidden_layers, num_output):
    self.setup_layer(num_input)
    self.setup_hidden(hidden_layers)
    self.setup_layer(num_output)

  def setup_layer(self, num_neurons):
    layer_neurons = np.array([])
    for x in range(num_neurons):
      input_weights = len(self.layers[-1]) if len(self.layers) > 0 else num_neurons
      layer_neurons = np.append(layer_neurons, self.neuronClass(input_weights))
    self.layers.append(layer_neurons)

  def setup_hidden(self, hidden_layers):
    if type(hidden_layers) is int:
      self.setup_layer(hidden_layers)
    else:
      for layer in hidden_layers:
        self.setup_layer(layer)

  def adjust_weights(self):
    self.learning.update(self.layers)

  def solve(self, input):
    previous_layer = np.array(input)
    for (i, layer) in enumerate(self.layers):
      new_previous_layer = np.array([])
      for neuron in layer:
        new_previous_layer = np.append(new_previous_layer, neuron.solve(previous_layer))
      print(i + 1, new_previous_layer)
      previous_layer = new_previous_layer
    self.adjust_weights()
    return previous_layer
