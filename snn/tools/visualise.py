import warnings
import matplotlib
import matplotlib.animation as animation
import networkx as nx
import numpy as np
import pylab
from matplotlib.pyplot import pause

pylab.ion()
warnings.filterwarnings("ignore",".*GUI is implemented.*")

test = [ [ { 'fired': True }, 1, 1 ], [ 1, 1, 1, 1, { 'fired': True }, 1 ], [ 1, 1, { 'fired': True } ], [ { 'fired': True } ] ]

G=nx.Graph()

pos = {}
labels = {}

def lengths(lists):
  return [len(list) for list in lists]

longest = max(lengths(test))
longest_mid_point = (longest - 1) / 2

for (i, layer) in enumerate(test):
  for (j, node) in enumerate(layer):
    node_name = str(i) + str(j)
    if i == 0: labels[node_name] = 'I'
    elif i == (len(test) - 1): labels[node_name] = 'O'
    else: labels[node_name] = 'H'

    if len(layer) == 0:
      pos[node_name] = np.array([ i, longest_mid_point ])
    else:
      current_layer_nodes = len(layer) - 1
      current_layer_mid_point = current_layer_nodes / 2
      modifier = longest_mid_point - current_layer_mid_point
      pos[node_name] = np.array([ i, j + modifier ])

for (i, layer) in enumerate(test):
  if i > 0:
    for (j, node) in enumerate(layer):
      node_name = str(i) + str(j)
      for (k, prev_node) in enumerate(test[i - 1]):
        prev_node_name = str(i - 1) + str (k)
        fired = type(prev_node) is dict and prev_node['fired']
        G.add_edge(prev_node_name, node_name, fired=fired)

fired_edges = [(u,v) for (u,v,d) in G.edges(data=True) if d['fired']]
edges = [(u,v) for (u,v,d) in G.edges(data=True) if not d['fired']]

def get_fig(i):
  color = 'red' if i % 2 else 'blue'
  nx.draw_networkx_nodes(G,pos,node_size=700,node_color='black')
  nx.draw_networkx_edges(G,pos,edgelist=fired_edges,edge_color=color)
  nx.draw_networkx_edges(G,pos,edgelist=edges,edge_color='black')
  nx.draw_networkx_labels(G,pos,font_size=12,font_family='sans-serif',font_color='white',labels=labels)

pylab.axis('off')
pylab.show()

num_plots = 30

for i in range(num_plots):
  get_fig(i)
  pylab.draw()
  try:
    pause(0.5);
  except:
    break;
