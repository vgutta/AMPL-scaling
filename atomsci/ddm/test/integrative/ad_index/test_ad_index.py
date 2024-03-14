#!/usr/bin/env python

import json
import numpy as np
import pandas as pd
import os
import sys
import glob
import pytest

import atomsci.ddm.pipeline.parameter_parser as parse
from atomsci.ddm.pipeline import model_pipeline as mp
from atomsci.ddm.pipeline import predict_from_model as pfm
import atomsci.ddm.utils.test_utils as tu

def clean():
    """
    Clean test files
    """
    if "output" not in os.listdir( os.path.dirname(__file__)):
        os.mkdir(tu.relative_to_file(__file__,"output"))
    for f in os.listdir(tu.relative_to_file(__file__,"./output")):
        if os.path.isfile(tu.relative_to_file(__file__,"./output/"+f)):
            os.remove(tu.relative_to_file(__file__,"./output/"+f))

@pytest.mark.basic
def test():
    """Test full model pipeline: Curate data, fit model, and predict property for new compounds"""

    # Clean
    # -----
    clean()

    # Run HyperOpt
    # ------------
    with open(tu.relative_to_file(__file__,"H1_RF.json"), "r") as f:
        hp_params = json.load(f)

    script_dir = parse.__file__.strip("parameter_parser.py").replace("/pipeline/", "")
    python_path = sys.executable
    hp_params["script_dir"] = script_dir
    hp_params["python_path"] = python_path

    params = parse.wrapper(hp_params)
    if not os.path.isfile(params.dataset_key):
        # reconstruct the absolute dir for the dataset key file (parent / test / test_datasets / file)
        params.dataset_key = os.path.join(params.script_dir, 'test', params.dataset_key.split(os.path.sep)[-2], params.dataset_key.split(os.path.sep)[-1])
    
    train_df = pd.read_csv(params.dataset_key)

    print(f"Train a RF models with ECFP")
    pl = mp.ModelPipeline(params)
    pl.train_model()

    print("Calculate AD index with the just trained model.")
    pred_df_mp = pl.predict_on_dataframe(train_df[:10], contains_responses=True, AD_method="z_score")

    assert("AD_index" in pred_df_mp.columns.values), 'Error: No AD_index column pred_df_mp'

    print("Calculate AD index with the saved model tarball file.")
    pred_df_file = pfm.predict_from_model_file(model_path=pl.params.model_tarball_path,
                                         input_df=train_df[:10],
                                         id_col="compound_id",
                                         smiles_col="base_rdkit_smiles",
                                         response_col="pKi_mean",
                                         dont_standardize=True,
                                         AD_method="z_score")
    assert("AD_index" in pred_df_file.columns.values), 'Error: No AD_index column in pred_df_file'

if __name__ == '__main__':
    test()
