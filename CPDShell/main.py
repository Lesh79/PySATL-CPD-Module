import tempfile
from pathlib import Path

from CPDShell.generator.generator import ScipyDatasetGenerator
from CPDShell.generator.saver import DatasetSaver
from CPDShell.shell import CPDShell

with tempfile.TemporaryDirectory() as tempdir:
    saver = DatasetSaver(Path(), True)
    generated = ScipyDatasetGenerator().generate_datasets(
        Path("tests/test_CPDShell/test_configs/test_config_1.yml"), saver
    )

    cpd = CPDShell(generated["exp"][0])
    res = cpd.run_cpd()
    res.visualize(True)
    print(res)
