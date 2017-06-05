from snn.learning.stdp import STDP

"""
Hedonistic variant of STDP Learning Method
"""
class HedonisticSTDP(STDP):
  def __init__(self):
    self.tau_c = 1
    self.tau_d = 0.2

  def calculate_c(self):
    self.c = (self.c / self.tau_c) + (STDP(self.tau) * (delta))
    return self.c

  def calculate_s(self):
    return self.c * self.d

  def calculate_d(self):
    self.d = (self.d / self.tau_d) + self.dopamine(t)

  def dopamine(self, t):
    return 1

  def update(self, layers):
    super(HedonisticSTDP, self).update(layers)
