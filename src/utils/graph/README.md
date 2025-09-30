# Graph Extension
This folder contains ``FlatlandGraph``, a translation of the flatland environment into a graph environment, which allows for simple path identification and serves as the basis for [negotiation](https://github.com/ozmanda/AI4REALNET-T3.4/tree/main/src/negotiation). The following classes are defined: 
1. **``FlatlandGraph``**: wraps the various functionalities into a single, accessible class. 
1. **``MultiDiGraphBuilder``**: constructs a multi-di graph (see [networkx documentation](https://networkx.org/documentation/stable/reference/classes/multidigraph.html)) from the flatland environment.
1. **``PathGenerator.py``**: contains functions for the generation of paths between arbitrary nodes in the graph.
1. **``ConflictPredictor``**: predicts conflicts on the basis of planned path, agent location and agent speed.