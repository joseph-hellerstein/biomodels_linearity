'''Constants used in the project'''

import numpy as np  # type: ignore
import os
import tellurium as te  # type: ignore

# Directories and paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
TEST_DIR = os.path.join(PROJECT_DIR, "tests")
REPO_DIR = os.path.dirname(PROJECT_DIR)
PSD_DATA_DIR = os.path.join(DATA_DIR, "psd_data")
BIOMODELS_DIR = os.path.join(REPO_DIR, "temp-biomodels", "final")
CALCULATED_ENTIMES_PATH = os.path.join(DATA_DIR, "biomodels_endtime.csv")
SERIALIZATION_DIR = os.path.join(PROJECT_DIR, "data", "serialize")
TIMECOURSE_SERIALIZATION_DIR = os.path.join(SERIALIZATION_DIR,
        "timecourse")
TIMECOURSE_ZIP_1000_PATH = os.path.join(TIMECOURSE_SERIALIZATION_DIR, "timecourse_1000.zip")
TIMECOURSE_ZIP_10000_PATH = os.path.join(TIMECOURSE_SERIALIZATION_DIR, "timecourse_10000.zip")
TIMECOURSE_ZIP_100000_PATH = os.path.join(TIMECOURSE_SERIALIZATION_DIR, "timecourse_100000.zip")
TIMECOURSE_ZIP_PATH = TIMECOURSE_ZIP_100000_PATH  # Default path to the timecourse zip file
PIECEWISE_PREDICTIONS_PATH = os.path.join(DATA_DIR, "piecewise_predictions.csv")
PAPER_DIR = os.path.join(PROJECT_DIR, "paper")
PLOT_DIR = os.path.join(PAPER_DIR, "plots")
NOTEBOOK_DIR = os.path.join(PROJECT_DIR, "notebooks")
CURATION_PATH = os.path.join(DATA_DIR, "curation.csv")
PIECEWISE_PREDICTIONS_MODEL_PATH = os.path.join(DATA_DIR, "piecewise_predictions_model.csv")

# Types
TYPE_ROADRUNNER = "tellurium.roadrunner.extended_roadrunner.ExtendedRoadRunner"
NULL_ROADRUNNER = te.loada("")

# Default values
START_TIME = 0.0
END_TIME = 10.0
NUM_POINT = 100000
SYSTEM_DISCOVERY_THRESHOLD = 0.01

# Diameter metrics
DIAMETER_IVP = "weighted_eigenvectors"
DIAMETER_MAX_CV = "max_cv"

# Columns
COL_A80 = "a80"  # Fraction of values with an accuracy of at least 80%.
COL_A90 = "a90"  # Fraction of values with an accuracy of at least 90%.
COL_A95 = "a95"  # Fraction of values with an accuracy of at least 95%.
COL_AUC = "auc"  # Area under the NLCurve computed from each row's changepoints.
COL_AGGREGATION_TYPE = "aggregation_type"  # model or species name
COL_AGGREGATION_TYPE_MODEL = "model"
COL_BOUNDARIES = "boundaries"  # List of segment boundaries calculated from the changepoints.
COL_CHANGEPOINTS = "changepoints"  # List of detected change points.
COL_COUNT = "count"  # Number of valid values used in calculating the statistics.
COL_CV = "cv"  # Coefficient of variation of the valid values used in calculating the statistics.
COL_CSV_FILE = "csv_file"  # Name of the CSV file containing the row of statistics.
COL_ENDTIME = "endtime"
COL_FILENAME = "filename"  # Name of the CSV file containing the row of statistics.
COL_FREQUENCIES = "frequencies"  # List of detected oscillation frequencies in Hz.
COL_IS_CHANGEPOINT_REMOVAL = "is_changepoint_removal"  # Whether change points that do not significantly improve the model were removed.
# FIXME: Change COL_MODEL_NAME to COL_SYSTEM_ID for consistency with other scripts.
COL_MODEL_NAME = "model_name"
COL_ENDTIME_SOURCE = "end_time_source"  # How end_time was determined (e.g. "reciprocal_min_eigenvalue", "default")
COL_INVALID_COUNT = "invalid_count"  # Number of invalid (sentinel -1) values excluded from aggregation.
COL_MAX_CHANGEPOINT = "max_changepoint"
COL_MEAN = "mean"  # Mean value of the valid values used in calculating the statistics.
COL_NUM_CHANGEPOINT = "num_changepoint"
COL_MAX_FRACTIONAL_REDUCTION = "max_fractional_reduction"
COL_NUM_REACTION = "num_reaction"
COL_PERTURBATION = "perturbation"  # Perturbation value fraction used in simulation
COL_LABEL = "label"  # Unique identifier for the row of statistics, e.g. model name or species name.
COL_MANYCP = "manycp"  # Whether the model has many change points.
COL_MAXCV = "max_cv"
COL_MEAN = "mean" # Mean value of the valid values used in calculating the statistics.
COL_MODEL_NUM = "model_num"  # Unique integer identifier for the model, e.g. 0, 1, 2, ...
COL_MAX = "max"  # Maximum value of the valid values used in calculating the statistics.
COL_MIN = "min"  # Minimum value of the valid values used in calculating the statistics.
COL_NUM_PERTURBATION = "num_perturbation"
COL_NUM_SPECIES = "num_species"
COL_NUM_POINT = "num_point"  # Number of points used in the simulation.
COL_P05 = "p05"  # 5th percentile of the valid values used in calculating the statistics.
COL_P10 = "p10"  # 10th percentile of the valid values used in calculating the statistics.
COL_P20 = "p20"  # 20th percentile of the valid values used in calculating the statistics.
COL_P25 = "p25"  # 25th percentile of the valid values used in calculating the statistics.
COL_P30 = "p30"  # 30th percentile of the valid values used in calculating the statistics.
COL_P50 = "p50"  # 50th percentile of the valid values used in calculating the statistics.
COL_P80 = "p80"  # 80th percentile of the valid values used in calculating the statistics.
COL_P90 = "p90"  # 90th percentile of the valid values used in calculating the statistics.
COL_P95 = "p95"  # 95th percentile of the valid values used in calculating the statistics.
COL_P99 = "p99"  # 99th percentile of the valid values used
COL_PERCENTILES = [COL_P05,COL_P10, COL_P20, COL_P25, COL_P30, COL_P50, COL_P80, COL_P90, COL_P95, COL_P99]
COL_REASON = "reason"  # Reason for exclusion of a model from analysis.
COL_REPEAT = "repeat"  # Which repetition of the model was used in the analysis.
COL_SPECIES_NAME = "species_name"
COL_START_TIME = "start_time"
COL_SYSTEM_ID = "system_id"  # Unique identifier for the system, e.g. model name or species name.
COL_STD = "std"  # Standard deviation of the valid values used in calculating the statistics.
COL_COEFFICIENT_THRESHOLD = "coefficient_threshold"  # Threshold used in SystemDiscovery
COLUMN_STATISTICS = [COL_MEAN, COL_MIN, COL_MAX, COL_COUNT, COL_INVALID_COUNT] + COL_PERCENTILES
COLUMN_ACCURACY_FRACTIONS = [COL_A80, COL_A90, COL_A95]
COLUMN_NAMES = [COL_MODEL_NAME, COL_MAXCV, COL_ENDTIME, COL_ENDTIME_SOURCE]
FILE_METADATA_KEY_2_COLUMN_METADATA_KEY_DCT = {
    "dummy": "dummy", # Used for testing purposes
    COL_NUM_POINT: "numpoint",
    COL_COEFFICIENT_THRESHOLD: "threshold",
    COL_IS_CHANGEPOINT_REMOVAL: "removal",
    COL_MAX_FRACTIONAL_REDUCTION: "maxreduction",
    COL_MANYCP: "manycp",
    COL_REPEAT: "repeat",
}
COLUMN_METADATA_KEY_2_FILE_METADATA_KEY_DCT = {v: k for k, v
        in FILE_METADATA_KEY_2_COLUMN_METADATA_KEY_DCT.items()}

# Symbolic values
ENDTIME_SOURCE_RECIROCAL_MIN_EIGENVALUE = "reciprocal_min_eigenvalue"
ENDTIME_SOURCE_SEDML = "sedml"
ENDTIME_SOURCE_STEADYSTATE = "steadystate"
ENDTIME_SOURCE_MAX_MEDIAN_CV = "max_median_cv"
ENDTIME_SOURCE_USER_SPECIFIED = "user_specified"

# Common default values
NULL_ARRAY = np.array([])

# Jacobian selection
JAC_FITTED = "fit_gershgorin" # Fit a Jacobian by using the timecourse for each row
JAC_MEDIAN = "median"  # Use the median Jacobian
JAC_FIRST = "first" # Use the first Jacobian

# Perturbation parameters for Jacobian fitting
PERTURBATION_VALUE_FRACTION = 0.0  # Perturb each point
PERTURBATION_SPECIES_FRACTION = 0.5  # Perturb this fraction of points

# Endtime fraction
ENDTIME_FRACTION_STEADYSTATE = 0.1  # Fraction of endtime to use for timecourse analysis
ENDTIME_FRACTION_MAXMEDIAN = 0.1  # Fraction of endtime to use for timecourse analysis
