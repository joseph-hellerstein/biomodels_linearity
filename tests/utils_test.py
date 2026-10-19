"""Commonly used code in tests."""

import src.constants as cn

import os


def makeBiomdName(model_num: int) -> str:
    """Construct the model name"""
    return f"BIOMD000000{model_num:04d}"

def makeBiomdPath(model_num: int) -> str:
    """Construct the file path for a BioModel SBML file given its model number."""
    model_name = makeBiomdName(model_num)
    return os.path.join(
        cn.BIOMODELS_DIR, model_name, f"{model_name}_url.xml"
    )