"""
world/model/layers/

Self-model layer implementations.

Each layer is an independent graph structure with its own node/edge types,
query interface, and update cadence. The SelfModel container (world/model/self_model.py)
holds one instance of each layer and routes SelfModelPatches to the right one.

Layers
------
factory_graph.py   FactoryGraph — directed graph of factory components.
                   The primary coordination data structure. Nodes are logical
                   factory units (production lines, mining sites, power grids,
                   etc.); edges are item flows between them.
"""
