from pathlib import Path

from CPDShell.Core.algorithms.bayesian_algorithm import BayesianAlgorithm
from CPDShell.Core.algorithms.BayesianCPD.detectors.drop_detector import DropDetector
from CPDShell.Core.algorithms.BayesianCPD.detectors.simple_detector import SimpleDetector
from CPDShell.Core.algorithms.BayesianCPD.hazards.constant_hazard import ConstantHazard
from CPDShell.Core.algorithms.BayesianCPD.likelihoods.gaussian_likelihood import GaussianLikelihood
from CPDShell.Core.algorithms.BayesianCPD.localizers.simple_localizer import SimpleLocalizer
from CPDShell.generator.generator import ScipyDatasetGenerator
from CPDShell.generator.saver import DatasetSaver
from CPDShell.shell import CPDShell

path_string = "tests/test_CPDShell/test_configs/test_config_exp.yml"
distributions_name = "exp"

saver = DatasetSaver(Path(), True)
generated = ScipyDatasetGenerator().generate_datasets(Path(path_string), saver)
data, actual_change_points = generated[distributions_name]

graph_cpd = CPDShell(data)
graph_cpd.scrubber.window_length = 150
graph_cpd.scrubber.movement_k = 2.0 / 3.0

print("Actual change points:", actual_change_points)

res_graph = graph_cpd.run_cpd()
res_graph.visualize(True)
print("Graph algorithm")
print(res_graph)

HAZARD_RATE = 200
LEARNING_WINDOW_SIZE = 30
THRESHOLD = 0.5
DROP_THRESHOLD = 0.7

constant_hazard = ConstantHazard(HAZARD_RATE)
gaussian_likelihood = GaussianLikelihood()

simple_detector = SimpleDetector(THRESHOLD)
drop_detector = DropDetector(DROP_THRESHOLD)

simple_localizer = SimpleLocalizer()

bayesian_algorithm = BayesianAlgorithm(
    learning_steps=LEARNING_WINDOW_SIZE,
    likelihood=gaussian_likelihood,
    hazard=constant_hazard,
    detector=drop_detector,
    localizer=simple_localizer,
)

bayesian_cpd = CPDShell(data, bayesian_algorithm)
bayesian_cpd.scrubber.window_length = 500
bayesian_cpd.scrubber.movement_k = 2.0 / 3.0

res_bayes = bayesian_cpd.run_cpd()
res_bayes.visualize(True)
print("Bayesian algorithm")
print(res_bayes)
