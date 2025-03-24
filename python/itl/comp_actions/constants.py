"""
Some constants shared across the module
"""
import networkx as nx

EPS = 1e-10                 # Value used for numerical stabilization
SR_THRES = 0.8              # Mismatch surprisal threshold
U_IN_PR = 0.99              # How much the agent values information provided by the user

# Connectivity graph that represents pairs of 3D inspection images to be cross-referenced
CON_GRAPH = nx.Graph()
for i in range(8):
    CON_GRAPH.add_edge(i, i+8)
    CON_GRAPH.add_edge(i+8, i+16)
    if i < 7:
        CON_GRAPH.add_edge(i, i+1)
        CON_GRAPH.add_edge(i+8, i+9)
        CON_GRAPH.add_edge(i+16, i+17)
    else:
        CON_GRAPH.add_edge(i, i-7)
        CON_GRAPH.add_edge(i+8, i+1)
        CON_GRAPH.add_edge(i+16, i+9)
# Index of viewpoints whose data (camera pose, visible points and their descriptors)
# will be stored in long-term memory; storing for all consumes too much space (storing
# all descriptors---even in reduced dimensionalities---would take too much)
STORE_VP_INDS = [0, 2, 4, 6, 16, 18, 20, 22]