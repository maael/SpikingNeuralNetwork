import numpy as np
from snn.learning.stdp import STDP

"""
Homeostatic variant of STDP Learning Method
"""
class HomeostaticSTDP(STDP):
  def __init__(self, options={}):
    super(self.__class__, self).__init__()
    np.seterr(all='ignore')
    self.setup_method(options)

  def set_self_options_or_default(self, name, options, default):
    value = default if name not in options else options[name]
    setattr(self, name, value)

  def setup_method(self, options):
    self.set_self_options_or_default('adjustment', options, 0.1)
    self.set_self_options_or_default('gaussian_mean', options, 0)
    self.set_self_options_or_default('gaussian_sd', options, 0.015)
    self.set_self_options_or_default('tau_stdp', options, 0.02)
    self.set_self_options_or_default('delta_t', options, 1)
    self.set_self_options_or_default('average_potentiation', options, 0)
    self.set_self_options_or_default('average_depression', options, 0)
    self.set_self_options_or_default('beta', options, (4 * (10 ^ -5)))
    self.set_self_options_or_default('gamma', options, (10 ^ -7))
    self.set_self_options_or_default('activity_goal', options, 1)
    self.set_self_options_or_default('is_setup', options, False)

  def setup_neuron(self, neuron):
    if not hasattr(neuron, 'firing_count'):
      setattr(neuron, 'firing_count', 0)
    if not hasattr(neuron, 'firing_rate'):
      setattr(neuron, 'firing_rate', 0)

  def calculate_neuron_firing_rate(self, neuron):
    if neuron.fired:
      neuron.firing_count = neuron.firing_count + 1
    neuron.firing_rate = neuron.firing_count / (self.time + 1)

  def update_potentiation(self, neuron):
    v = np.random.normal(self.gaussian_mean, self.gaussian_sd)
    neuron.weights = np.add(neuron.weights, np.multiply(np.add(self.average_potentiation, np.multiply(v, neuron.weights)), np.exp(-1/self.tau_stdp)))

  def update_depression(self, neuron):
    v = np.random.normal(self.gaussian_mean, self.gaussian_sd)
    neuron.weights = np.add(neuron.weights, np.multiply(np.add(np.multiply(-self.average_depression, neuron.weights), np.multiply(v, neuron.weights)), np.exp(1/self.tau_stdp)))

  def activity_dependent_scaling(self, neuron):
    neuron.weights = np.multiply(np.multiply(self.beta, neuron.weights), (self.activity_goal - neuron.firing_rate))

  def update_weights(self, neuron):
    if neuron.refractoryTime is 0:
      self.update_potentiation(neuron)
    else:
      self.update_depression(neuron)

  def update(self, layers):
    for layer in layers:
      for neuron in layer:
        if not self.is_setup:
          self.setup_neuron(neuron)
        self.calculate_neuron_firing_rate(neuron)
        self.update_weights(neuron)
        self.activity_dependent_scaling(neuron)
    if not self.is_setup: self.is_setup = True
    self.time = self.time + 1
