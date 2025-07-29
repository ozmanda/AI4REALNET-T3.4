# Flatland Railway Extension
This extension contains function which exten the flatland rail simulator, and are partially developed by [Adrian Egli](https://github.com/aiAdrian), taken from the [flatland_railway_extension](https://github.com/aiAdrian/flatland_railway_extension/tree/master?tab=readme-ov-file) repo, and extended by [me](https://github.com/ozmanda). 

## 1. Flatland MultiDiGraph
The original graph implementation by Adrian resulted in a graph in which each switch could have up to four nodes, making pathfinding within the graph very efficient. For message propagation, however, it is essential for spatially "close" nodes to be near each other in the graph. Similarly to ``FlatlandGraphBuilder.py``, the ``MultiDiGraphBuilder.py`` transforms a flatland ``RailEnv`` into a ``MultiDiGraph`` from the ``networkx`` package. 

![Image of translation from DiGraph to MultiDiGraph](imgs/DiGraph-MultiDiGraph.png)

The graph contains the following network information in the form of node and edge attributes: (WIP)
- **Node Attributes**
    - ``dead_end``: indicates whether the node is a dead end

- **Edge Attributes** 
    - ``rail_ID``: the ID of the rail cluster the edge belongs to
    - ``out_direction``: the direction of travel from the origin node
    - ``in_direction``: the direction of travel to the destination node
    - ``length``: the length of the edge (number of cells)
    - ``resources``: list of cells in the edge including their travel direction tuple (out, in)
    - ``max_speed``: max speed along the edge (WIP)