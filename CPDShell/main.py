from pathlib import Path

from CPDShell.Core.algorithms.graph_algorithm import GraphAlgorithm
from CPDShell.Core.cpd_core import CPDCore, Scrubber
from CPDShell.Core.scenario import Scenario
from CPDShell.Shell import LabeledCPData

scenario = Scenario(10, True)
scrubber = Scrubber(scenario, [1, 1.2, 12, 13, 11, 12, 1, 1, 1, 1, 1])
core = CPDCore(scrubber, GraphAlgorithm(1, 2))

data = LabeledCPData.generate_cp_dataset(
    Path("tests/test_CPDShell/test_configs/test_config_1.yml"), to_save=True, output_directory=Path("data")
)
for labeled_data in data:
    print(labeled_data)
