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
        "--version",
        default="version_2",
        choices=["version_1", "version_2"],
        help="Version to use",
    )

    parser.add_argument(
        "--prediction-task",
        default="Length of Stay",
        choices=["Mortality", "Length of Stay", "Readmission", "Phenotype"],
        help="Prediction task to do",
    )

    parser.add_argument(
        "--Length-of-Stay",
        default="Length of Stay ge 3",
        choices=["Length of Stay ge 3", "Length of Stay ge 7", "Custom"],
        help="Prediction task to do",
    )

    parser.add_argument(
        "--Readmission",
        default="30 Day Readmission",
        choices=[
            "30 Day Readmission",
            "60 Day Readmission",
            "90 Day Readmission",
            "120 Day Readmission",
            "Custom",
        ],
        help="Readmission after (in days)",
    )

    parser.add_argument(
        "--Phenotype",
        default="Heart Failure in 30 days",
        choices=[
            "Heart Failure in 30 days",
            "CAD in 30 days",
            "CKD in 30 days",
            "COPD in 30 days",
        ],
        help="Phenotype",
    )

    parser.add_argument(
        "--data",
        default="ICU",
        choices=["ICU", "Non-ICU"],
        help="if you want to work with ICU or Non-ICU data",
    )

    parser.add_argument(
        "--no-diagnosis", action="store_true", help="Disables diagnosis"
    )

    parser.add_argument(
        "--no-output-events", action="store_true", help="Disables diagnosis"
    )

    parser.add_argument("--no-labs", action="store_true", help="Disables diagnosis")

    parser.add_argument(
        "--no-procedures", action="store_true", help="Disables diagnosis"
    )

    parser.add_argument(
        "--no-medications", action="store_true", help="Disables medications"
    )

    parser.add_argument(
        "--Disease-filter",
        default="No Disease Filter",
        choices=["No Disease Filter", "Heart Failure", "CKD", "CAD", "COPD"],
        help="if you want to perform choosen prediction task for a specific disease",
    )

    parser.add_argument(
        "--group-ICD-10-DIAG-codes",
        default="Convert ICD-9 to ICD-10 and group ICD-10 codes",
        choices=[
            "Keep both ICD-9 and ICD-10 codes",
            "Convert ICD-9 to ICD-10 codes",
            "Convert ICD-9 to ICD-10 and group ICD-10 codes",
        ],
        help="Do you want to group ICD 10 DIAG codes",
    )

    parser.add_argument(
        "--Non-propietary-names",
        default="yes",
        choices=["yes", "no"],
        help="Do you want to group Medication codes to use Non propietary names",
    )

    parser.add_argument(
        "--ICD-codes-for-Procedures",
        default="ICD-10",
        choices=["ICD-9 and ICD-10", "ICD-10"],
        help="Which ICD codes for Procedures you want to keep in data?",
    )

    parser.add_argument(
        "--diag_features",
        default="yes",
        choices=["yes", "no"],
        help="Do you want to do Feature Selection for Diagnosis \n (If yes, please edit list of codes in ./data/summary/diag_features.csv)",
    )

    parser.add_argument(
        "--med_features",
        default="yes",
        choices=["yes", "no"],
        help="Do you want to do Feature Selection for Medication \n (If yes, please edit list of codes in ./data/summary/med_features.csv)",
    )

    parser.add_argument(
        "--proc_features",
        default="yes",
        choices=["yes", "no"],
        help="Do you want to do Feature Selection for Procedures \n (If yes, please edit list of codes in ./data/summary/proc_features.csv)",
    )

    parser.add_argument(
        "--out_features",
        default="yes",
        choices=["yes", "no"],
        help="Do you want to do Feature Selection for Chart events \n (If yes, please edit list of codes in ./data/summary/out_features.csv)",
    )

    parser.add_argument(
        "--chart_features",
        default="yes",
        choices=["yes", "no"],
        help="Do you want to do Feature Selection for Output event \n (If yes, please edit list of codes in ./data/summary/chart_features.csv)",
    )

    parser.add_argument(
        "--lab_features",
        default="yes",
        choices=["yes", "no"],
        help="Do you want to do Feature Selection for Labs \n (If yes, please edit list of codes in ./data/summary/lab_features.csv)",
    )

    parser.add_argument(
        "--Outlier-removal",
        default="No outlier detection",
        choices=[
            "No outlier detection",
            "Impute Outlier (default:98)",
            "Remove outliers (default:98)",
        ],
        help="Outlier removal in values of chart/lab events?",
    )

    parser.add_argument(
        "--include1",
        default="First 72 hours",
        choices=["First 72 hours", "First 48 hours", "First 24 hours", "Custom"],
        help="Length of data to be included for time-series prediction",
    )

    parser.add_argument(
        "--include2",
        default="First 24 hours",
        choices=["First 12 hours", "First 24 hours", "Custom"],
        help="Length of data to be included for time-series prediction",
    )

    parser.add_argument(
        "--include3",
        default="First 72 hours",
        choices=["First 72 hours", "First 48 hours", "First 24 hours", "Custom"],
        help="Length of data to be included for time-series prediction",
    )

    parser.add_argument(
        "--time-bucket",
        default="1 hour",
        choices=["1 hour", "2 hour", "3 hour", "4 hour", "5 hour", "Custom"],
        help="What time bucket size you want to choose ?",
    )

    parser.add_argument(
        "--impute",
        default="No Imputation",
        choices=["No Imputation", "forward fill and mean", "forward fill and median"],
        help="Do you want to forward fill and mean or median impute lab/chart values to form continuous data signal?",
    )

    parser.add_argument(
        "--window",
        default="2 hours",
        choices=["2 hours", "4 hours", "6 hours", "8 hours", "Custom"],
        help="If you have choosen mortality prediction task, then what prediction window length you want to keep?",
    )

    parser.add_argument(
        "--model-type",
        default="Time-series LSTM",
        choices=[
            "Time-series LSTM",
            "Time-series CNN",
            "Hybrid LSTM",
            "Hybrid CNN",
            "Transformer",
        ],
        help="Choose a model",
    )

    parser.add_argument(
        "--cross-validation",
        default="5-fold CV",
        choices=["No CV", "5-fold CV", "10-fold CV"],
        help="option for cross-validation",
    )
    parser.add_argument(
        "--oversampling",
        default="True",
        choices=["True", "False"],
        help="Do you want to do oversampling for minority calss ?",
    )

    args = parser.parse_args()

    return args


def main(args):
    breakpoint()

    disease_label = ""
    label = args.prediction_task
    time = 0

    if args.prediction_task == "Readmission":
        if args.Readmission == "Custom":
            time = 0  # TODO
        else:
            time = int(args.Readmission.split()[0])
    elif args.prediction_task == "Length of Stay":
        if args.Length_of_Stay == "Custom":
            time = 0  # TODO
        else:
            time = int(args.Length_of_Stay.split()[4])

    if args.prediction_task == "Phenotype":
        if args.Phenotype == "Heart Failure in 30 days":
            label = "Readmission"
            time = 30
            disease_label = "I50"
        elif args.Phenotype == "CAD in 30 days":
            label = "Readmission"
            time = 30
            disease_label = "I25"
        elif args.Phenotype == "CKD in 30 days":
            label = "Readmission"
            time = 30
            disease_label = "N18"
        elif args.Phenotype == "COPD in 30 days":
            label = "Readmission"
            time = 30
            disease_label = "J44"

    data_icu = args.data == "ICU"
    data_mort = label == "Mortality"
    data_admn = label == "Readmission"
    data_los = label == "Length of Stay"

    if args.Disease_filter == "Heart Failure":
        icd_code = "I50"
    elif args.Disease_filter == "CKD":
        icd_code = "N18"
    elif args.Disease_filter == "COPD":
        icd_code = "J44"
    elif args.Disease_filter == "CAD":
        icd_code = "I25"
    else:
        icd_code = "No Disease Filter"

    root_dir = os.path.dirname(os.path.abspath("mainPipeline.py"))
    if args.version == "version_1":
        version_path = "mimiciv/1.0"
        cohort_output = day_intervals_cohort.extract_data(
            args.data, label, time, icd_code, root_dir, disease_label
        )
    elif args.version == "version_2":
        version_path = "mimiciv/2.0"
        cohort_output = day_intervals_cohort_v2.extract_data(
            args.data, label, time, icd_code, root_dir, disease_label
        )

    if data_icu:
        diag_flag = not args.no_diagnosis
        out_flag = not args.no_output_events
        chart_flag = not args.no_labs
        proc_flag = not args.no_procedures
        med_flag = not args.no_medications
        feature_icu(
            cohort_output,
            version_path,
            diag_flag,
            out_flag,
            chart_flag,
            proc_flag,
            med_flag,
        )
    else:
        diag_flag = not args.no_diagnosis
        lab_flag = not args.no_labs
        proc_flag = not args.no_procedures
        med_flag = not args.no_medications
        feature_nonicu(
            cohort_output, version_path, diag_flag, lab_flag, proc_flag, med_flag
        )

    group_diag = False
    group_med = False
    group_proc = False
    if data_icu:
        if diag_flag:
            group_diag = args.group_ICD_10_DIAG_codes
        preprocess_features_icu(
            cohort_output, diag_flag, group_diag, False, False, False, 0, 0
        )
    else:
        if diag_flag:
            group_diag = args.group_ICD_10_DIAG_codes
        if med_flag:
            group_med = args.Non_propietary_names
        if proc_flag:
            group_proc = args.ICD_codes_for_Procedures
        preprocess_features_hosp(
            cohort_output,
            diag_flag,
            proc_flag,
            med_flag,
            False,
            group_diag,
            group_med,
            group_proc,
            False,
            False,
            0,
            0,
        )

    if data_icu:
        generate_summary_icu(diag_flag, proc_flag, med_flag, out_flag, chart_flag)
    else:
        generate_summary_hosp(diag_flag, proc_flag, med_flag, lab_flag)

    if data_icu:
        if diag_flag:
            select_diag = args.diag_features == "Yes"
        if med_flag:
            select_med = args.med_features == "Yes"
        if proc_flag:
            select_proc = args.proc_features == "Yes"
        if out_flag:
            select_out = args.out_features == "Yes"
        if chart_flag:
            select_chart = args.chart_features == "Yes"
        features_selection_icu(
            cohort_output,
            diag_flag,
            proc_flag,
            med_flag,
            out_flag,
            chart_flag,
            select_diag,
            select_med,
            select_proc,
            select_out,
            select_chart,
        )
    else:
        if diag_flag:
            select_diag = args.diag_features == "Yes"
        if med_flag:
            select_med = args.med_features == "Yes"
        if proc_flag:
            select_proc = args.proc_features == "Yes"
        if lab_flag:
            select_lab = args.lab_features == "Yes"
        features_selection_hosp(
            cohort_output,
            diag_flag,
            proc_flag,
            med_flag,
            lab_flag,
            select_diag,
            select_med,
            select_proc,
            select_lab,
        )

    thresh = 0
    if data_icu:
        if chart_flag:
            clean_chart = args.Outlier_removal != "No outlier detection"
            impute_outlier_chart = args.Outlier_removal == "Impute Outlier (default:98)"
            thresh = 98
            left_thresh = 0
        preprocess_features_icu(
            cohort_output,
            False,
            False,
            chart_flag,
            clean_chart,
            impute_outlier_chart,
            thresh,
            left_thresh,
        )
    else:
        if lab_flag:
            clean_lab = args.Outlier_removal != "No outlier detection"
            impute_outlier = args.Outlier_removal == "Impute Outlier (default:98)"
            thresh = 98
            left_thresh = 0
        preprocess_features_hosp(
            cohort_output,
            False,
            False,
            False,
            lab_flag,
            False,
            False,
            False,
            clean_lab,
            impute_outlier,
            thresh,
            left_thresh,
        )

    if args.window == "Custom":
        predW = ""  # TODO
    else:
        predW = int(args.window[0].strip())
    if args.time_bucket == "Custom":
        bucket = ""  # TODO
    else:
        bucket = int(args.time_bucket[0].strip())

    if data_mort:
        if args.include1 == "Custom":
            include = ""  # TODO
        else:
            include = int(args.include1.split()[1])
    elif data_admn:
        if args.include2 == "Custom":
            include = ""  # TODO
        else:
            include = int(args.include2.split()[1])
    elif data_los:
        if args.include2 == "Custom":
            include = ""  # TODO
        else:
            include = int(args.include3.split()[1])
    if args.impute == "forward fill and mean":
        impute = "Mean"
    elif args.impute == "forward fill and median":
        impute = "Median"
    else:
        impute = False

    if data_icu:
        gen = data_generation_icu.Generator(
            cohort_output,
            data_mort,
            data_admn,
            data_los,
            diag_flag,
            proc_flag,
            out_flag,
            chart_flag,
            med_flag,
            impute,
            include,
            bucket,
            predW,
        )
    else:
        gen = data_generation.Generator(
            cohort_output,
            data_mort,
            data_admn,
            data_los,
            diag_flag,
            lab_flag,
            proc_flag,
            med_flag,
            impute,
            include,
            bucket,
            predW,
        )

    if args.cross_validation == "No CV":
        cv = 0
    elif args.cross_validation == "5-fold CV":
        cv = int(5)
    elif args.cross_validation == "10-fold CV":
        cv = int(10)

    if data_icu:
        if args.model_type == "Transformer":
            model = dl_train.DL_models2(
                data_icu,
                diag_flag,
                proc_flag,
                out_flag,
                chart_flag,
                med_flag,
                False,
                args.model_type,
                cv,
                oversampling=args.oversampling == "True",
                model_name="attn_icu_read",
                train=True,
            )
        else:
            model = dl_train.DL_models(
                data_icu,
                diag_flag,
                proc_flag,
                out_flag,
                chart_flag,
                med_flag,
                False,
                args.model_type,
                cv,
                oversampling=args.oversampling == "True",
                model_name="attn_icu_read",
                train=True,
            )
    else:
        if args.model_type == "Transformer":
            model = dl_train.DL_models2(
                data_icu,
                diag_flag,
                proc_flag,
                False,
                False,
                med_flag,
                lab_flag,
                args.model_type,
                cv,
                oversampling=args.oversampling == "True",
                model_name="attn_icu_read",
                train=True,
            )
        else:
            model = dl_train.DL_models(
                data_icu,
                diag_flag,
                proc_flag,
                False,
                False,
                med_flag,
                lab_flag,
                args.model_type,
                cv,
                oversampling=args.oversampling == "True",
                model_name="attn_icu_read",
                train=True,
            )


def train(args):
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
        model = dl_train.DL_model2(
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
    # train(args)
