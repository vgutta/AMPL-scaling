#!/usr/bin/env python

import json
import numpy as np
import pandas as pd
import os
import sys
import glob

import atomsci.ddm.pipeline.parameter_parser as parse
import pytest
import atomsci.ddm.utils.test_utils as tu

from pathlib import Path


def clean():
    """Clean test files"""
    if "output" not in os.listdir(Path(__file__).parent):
        os.mkdir(tu.relative_to_file(__file__, "output"))
    for f in os.listdir(tu.relative_to_file(__file__, "output")):
        if os.path.isfile(tu.relative_to_file(__file__, "output/" + f)):
            os.remove(tu.relative_to_file(__file__, "output/" + f))
    if "tmp" not in os.listdir():
        os.mkdir(tu.relative_to_file(__file__, "tmp"))
    for f in os.listdir(tu.relative_to_file(__file__, "tmp")):
        if os.path.isfile(tu.relative_to_file(__file__, "tmp/") + f):
            os.remove(tu.relative_to_file(__file__, "tmp/") + f)


@pytest.mark.basic
def test():
    """Test full model pipeline: Curate data, fit model, and predict property for new compounds"""

    # Clean
    # -----
    clean()

    # Run HyperOpt
    # ------------
    with open(tu.relative_to_file(__file__, "H1_RF_hyperopt.json"), "r") as f:
        hp_params = json.load(f)

    script_dir = parse.__file__.strip("parameter_parser.py").replace("/pipeline/", "")
    python_path = sys.executable
    hp_params["script_dir"] = script_dir
    hp_params["python_path"] = python_path
    hp_params["result_dir"] = tu.relative_to_file(__file__, hp_params["result_dir"])

    params = parse.wrapper(hp_params)
    if not os.path.isfile(params.dataset_key):
        hp_params["dataset_key"] = os.path.join(script_dir, hp_params["dataset_key"])

    with open(tu.relative_to_file(__file__, "H1_RF_hyperopt_temp.json"), "w") as f:
        json.dump(hp_params, f, indent=4)

    run_cmd = f"{python_path} {script_dir}/utils/hyperparam_search_wrapper.py --config_file ./H1_RF_hyperopt_temp.json"
    os.system(run_cmd)

    # check results
    # -------------
    perf_table = glob.glob(tu.relative_to_file(__file__, "./output/performance*"))
    best_model = glob.glob(tu.relative_to_file(__file__, "./output/best*"))

    assert len(perf_table) == 1, "Error: No performance table returned."
    assert len(best_model) == 1, "Error: No best model saved"
    perf_df = pd.read_csv(perf_table[0])
    assert len(perf_df) == 10, "Error: Size of performance table WRONG."


if __name__ == "__main__":
    test()
