import csv
from pathlib import Path

import yaml

from CPDShell.generator.generator import ScipyDatasetGenerator
from CPDShell.generator.saver import DatasetSaver

SAMPLE_SIZE = 500
CP_LOCATION = 250

NUM_OF_SAMPLES = 1000

DIR_PATH = "/experiment/stage_1/"
CONFIG_NAME = "config.yml"

WORKING_DIR = Path()


class VerboseSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def distribution_template(distribution_type, length, parameters):
    return {"type": distribution_type, "length": length, "parameters": parameters}


def add_distribution(distributions, distribution_type, parameters, length=CP_LOCATION):
    distributions[distribution_type] = distribution_template(distribution_type, length, parameters)


distributions_left = {}
add_distribution(distributions_left, "normal", {"mean": 0.0, "variance": 1.0})
add_distribution(distributions_left, "exponential", {"rate": 1.0})
add_distribution(distributions_left, "uniform", {"min": 0.0, "max": 1.0})
add_distribution(distributions_left, "weibull", {"shape": 1.0, "scale": 0.5})
add_distribution(distributions_left, "beta", {"alpha": 0.5, "beta": 0.5})

distributions_right = {}
add_distribution(distributions_right, "normal", {"mean": 10.0, "variance": 5.0})
add_distribution(distributions_right, "exponential", {"rate": 5.0})
add_distribution(distributions_right, "uniform", {"min": 1.0, "max": 4.0})
add_distribution(distributions_right, "weibull", {"shape": 1.0, "scale": 5.0})
add_distribution(distributions_right, "beta", {"alpha": 5.0, "beta": 5.0})

distributions_no_cp = {}
add_distribution(distributions_no_cp, "normal", {"mean": 0.0, "variance": 1.0}, SAMPLE_SIZE)
add_distribution(distributions_no_cp, "exponential", {"rate": 1.0}, SAMPLE_SIZE)
add_distribution(distributions_no_cp, "uniform", {"min": 0.0, "max": 1.0}, SAMPLE_SIZE)
add_distribution(distributions_no_cp, "weibull", {"shape": 1.0, "scale": 0.5}, SAMPLE_SIZE)
add_distribution(distributions_no_cp, "beta", {"alpha": 0.5, "beta": 0.5}, SAMPLE_SIZE)


def generate_configs(distributions):
    distribution_types = []
    distributions_l, distributions_r, distributions_without_cp = distributions
    for distribution in distributions_l:
        distribution_types.append(distribution)

    names = []
    for distribution_type_l in distribution_types:
        for distribution_type_r in distribution_types:
            name = distribution_type_l + "-" + distribution_type_r
            names.append(name)

            config = [
                {
                    "name": name,
                    "distributions": [distributions_l[distribution_type_l], distributions_r[distribution_type_r]],
                }
            ]

            Path(WORKING_DIR / f"experiment/stage_1/{name}/").mkdir(parents=True, exist_ok=True)

            with open(WORKING_DIR / f"experiment/stage_1/{name}/config.yaml", "w") as outfile:
                yaml.dump(config, outfile, default_flow_style=False, sort_keys=False, Dumper=VerboseSafeDumper)

    for name in distribution_types:
        names.append(name)

        config = [
            {
                "name": name,
                "distributions": [
                    distributions_without_cp[name],
                ],
            }
        ]

        Path(WORKING_DIR / f"experiment/stage_1/{name}/").mkdir(parents=True, exist_ok=True)

        with open(WORKING_DIR / f"experiment/stage_1/{name}/config.yaml", "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False, sort_keys=False, Dumper=VerboseSafeDumper)

    return names


Path(WORKING_DIR / "experiment/stage_1/").mkdir(parents=True, exist_ok=True)
generated_names = generate_configs((distributions_left, distributions_right, distributions_no_cp))

with open(WORKING_DIR / "experiment/stage_1/experiment_description", "w", newline="") as f:
    write = csv.writer(f)

    write.writerow(["name", "samples_num"])
    samples_description = [[s, str(NUM_OF_SAMPLES)] for s in generated_names]
    write.writerows(samples_description)

for generated_name in generated_names:
    Path(WORKING_DIR / f"experiment/stage_1/{generated_name}/").mkdir(parents=True, exist_ok=True)

    for sample_num in range(NUM_OF_SAMPLES):
        print(f"Name: {generated_name}. Sample num: {sample_num}")
        Path(WORKING_DIR / f"experiment/stage_1/{generated_name}/sample_{sample_num}/").mkdir(
            parents=True, exist_ok=True
        )
        saver = DatasetSaver(WORKING_DIR / f"experiment/stage_1/{generated_name}/sample_{sample_num}/", True)
        generated = ScipyDatasetGenerator().generate_datasets(
            Path(WORKING_DIR / f"experiment/stage_1/{generated_name}/config.yaml"), saver
        )

        Path(
            WORKING_DIR / f"experiment/stage_1/{generated_name}/sample_{sample_num}/{generated_name}/sample.png"
        ).unlink(missing_ok=True)
