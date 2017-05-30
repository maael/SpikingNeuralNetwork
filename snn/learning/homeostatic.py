from snn.learning.method import LearningMethod

"""
Homeostatic variant of STDP Learning Method
"""
class HomeostaticSTDP(LearningMethod):
  def __init__(self):
    self.adjustment = 0.1

  def update(self, layers):
    neuron.weights = neuron.weights
