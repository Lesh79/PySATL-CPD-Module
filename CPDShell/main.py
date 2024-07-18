from CPDShell.Core.algorithms.graph_algorithm import GraphAlgorithm
from CPDShell.Core.CPDCore import CPDCore, Scrubber
from CPDShell.Core.scenario import Scenario

scenario = Scenario(10, True)
scrubber = Scrubber(scenario, [1, 1.2, 12, 13, 11, 12, 1, 1, 1, 1, 1])
core = CPDCore(scrubber, GraphAlgorithm(1, 2))
print(core.run())
