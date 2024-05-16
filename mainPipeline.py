import sys
import os

module_path = "preprocessing/day_intervals_preproc"
if module_path not in sys.path:
    sys.path.append(module_path)

module_path = "utils"
if module_path not in sys.path:
    sys.path.append(module_path)

module_path = "preprocessing/hosp_module_preproc"
if module_path not in sys.path:
    sys.path.append(module_path)

module_path = "model"
if module_path not in sys.path:
    sys.path.append(module_path)

import day_intervals_cohort, day_intervals_cohort_v2, data_generation_icu, data_generation, evaluation, feature_selection_hosp, ml_models, dl_train, tokenization, behrt_train, feature_selection_icu, fairness, callibrate_output
from day_intervals_cohort import *
from day_intervals_cohort_v2 import *
from feature_selection_hosp import *
from ml_models import *
from dl_train import *
from tokenization import *
from behrt_train import *
from feature_selection_icu import *

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prediction-task",
        default="Mortality",
        choices=["Mortality", "Length_of_Stay", "Readmission", "Phenotype"],
        help="Prediction task to do",
    )

    parser.add_argument(
        "--non-icu-data",
        action="store_true",
        help="Disables the use of icu data",
    )

    parser.add_argument(
        "--specific-disease",
        default="No Disease Filter",
        choices=["No_Disease_Filter", "Heart_Failure", "CKD", "CAD", "COPD"],
        help="Choose a specific disease for the prediction task",
    )

    parser.add_argument(
        "--model-type",
        default="Time-series_LSTM",
        choices=[
            "Time-series_LSTM",
            "Time-series_CNN",
            "Hybrid_LSTM",
            "Hybrid_CNN",
            "Transformer",
        ],
        help="Choose a model",
    )

    args = parser.parse_args()

    return args


def main(args):
    if args.model_type == "Transformer":
        model = dl_train.DL_models2(
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            args.model_type,
            0,
            False,
            "attn_icu_read",
            train=True,
        )
    else:
        model = dl_train.DL_models(
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            args.model_type,
            0,
            False,
            "attn_icu_read",
            train=True,
        )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
