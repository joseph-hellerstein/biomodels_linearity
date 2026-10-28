"""
Classify reaction rate laws for BioModels SBML models using the SBMLKinetics 2DK scheme.

Iterates over BioModels models in cn.BIOMODELS_DIR within a model-number range,
classifies each reaction's kinetic law into one of ten mutually exclusive
kinetics types (ZERO, UNDR, UNMO, BIDR, BIMO, MM, MMCAT, HILL, FR, NA), and
writes the per-model counts to a CSV file.

Classification follows the SBMLKinetics reference implementation: model
function definitions are expanded into each formula, the formula is simplified
with sympy, and the type checks run in order with the first match winning.
Reactions without a kinetic law are counted separately and not classified.

Usage:
    source activate.sh
    python scripts/rate_law_statistics.py --first_model_num 1 --last_model_num 56 [--output FILE]

Arguments:
    --first_model_num   First BioModel number to process (inclusive, default 0).
    --last_model_num    Last BioModel number to process (inclusive, default: all models).
    --output            Path for the output CSV file (default: cn.DATA_DIR/rate_law_statistics.csv).
"""
import src.constants as cn  # type: ignore
from src.biomodels_iterator import BiomodelsIterator  # type: ignore

import argparse
import os
import sys

import pandas as pd  # type: ignore
from sympy import simplify  # type: ignore

import libsbml  # type: ignore
from SBMLKinetics.common.function_definition import FunctionDefinition  # type: ignore
from SBMLKinetics.common.reaction import Reaction  # type: ignore

# (label, KineticLaw method) pairs in the reference implementation's order.
K_TYPE_CHECKS = [
    ("ZERO", "isZerothOrder"),
    ("UNDR", "isUNDR"),
    ("UNMO", "isUNMO"),
    ("BIDR", "isBIDR"),
    ("BIMO", "isBIMO"),
    ("MM", "isMM"),
    ("MMCAT", "isMMcat"),
    ("HILL", "isHill"),
    ("FR", "isFraction"),
]
NA = "NA"
K_TYPE_LABELS = [label for label, _ in K_TYPE_CHECKS] + [NA]
OUTPUT_FILENAME = "rate_law_statistics.csv"
OUTPUT_PATH = os.path.join(cn.DATA_DIR, OUTPUT_FILENAME)
CSV_COLUMNS = [cn.COL_MODEL_NUM, cn.COL_MODEL_NAME, cn.COL_NUM_SPECIES,
        cn.COL_NUM_REACTION, "num_reaction_with_kinetic_law",
        "num_reaction_without_kinetic_law"] + K_TYPE_LABELS


def readModel(sbmlPath: str) -> libsbml.Model:
    """Read an SBML file and return its libsbml Model.

    Exits with a message on stderr if the file cannot be parsed.
    """
    reader = libsbml.SBMLReader()
    document = reader.readSBML(sbmlPath)
    model = document.getModel() if document is not None else None
    if model is None:
        print(f"Error: could not parse SBML file: {sbmlPath}", file=sys.stderr)
        sys.exit(1)
    return model


def getFunctionDefinitions(model: libsbml.Model) -> list[FunctionDefinition]:
    """Wrap all of the model's function definitions for SBMLKinetics."""
    return [
        FunctionDefinition(model.getFunctionDefinition(n))
        for n in range(model.getNumFunctionDefinitions())
    ]


def classifyReaction(
    reaction: Reaction,
    function_definitions: list[FunctionDefinition],
    species_set: set[str],
) -> str:
    """Classify one reaction's kinetic law and return a K-type label.

    Runs the SBMLKinetics type checks in order and returns the first match,
    or NA if none apply. A check that raises on unusual kinetics is treated
    as a non-match so one odd reaction cannot abort the whole model.
    """
    reaction.kinetic_law.mkSymbolExpression(function_definitions)

    reactant_list = [r.getSpecies() for r in reaction.reactants]
    product_list = [p.getSpecies() for p in reaction.products]

    kinetics = reaction.kinetic_law.expanded_formula
    try:
        kinetics_sim = str(simplify(kinetics))
    except Exception:
        kinetics_sim = kinetics

    ids_list = list(dict.fromkeys(reaction.kinetic_law.symbols))
    species_in_kinetic_law = [i for i in ids_list if i in species_set]
    parameters_in_kinetic_law = [i for i in ids_list if i not in species_set]
    # Include reactants and products so denominator checks see all involved ids.
    ids_list = list(dict.fromkeys(ids_list + reactant_list + product_list))

    kwargs = {
        "kinetics": kinetics,
        "kinetics_sim": kinetics_sim,
        "reactant_list": reactant_list,
        "product_list": product_list,
        "species_in_kinetic_law": species_in_kinetic_law,
        "parameters_in_kinetic_law": parameters_in_kinetic_law,
        "ids_list": ids_list,
    }

    for label, method_name in K_TYPE_CHECKS:
        check = getattr(reaction.kinetic_law, method_name)
        try:
            if check(**kwargs):
                return label
        except Exception:
            continue
    return NA


def analyzeModel(sbmlPath: str) -> dict:
    """Parse sbmlPath and return rate-law statistics for the model."""
    model = readModel(sbmlPath)
    function_definitions = getFunctionDefinitions(model)
    species_list = [model.getSpecies(n).getId() for n in range(model.getNumSpecies())]
    species_set = set(species_list)

    counts = {label: 0 for label, _ in K_TYPE_CHECKS}
    counts[NA] = 0
    no_kinetic_law = 0
    for n in range(model.getNumReactions()):
        libsbml_reaction = model.getReaction(n)
        reaction_id = libsbml_reaction.getId()
        if libsbml_reaction.getKineticLaw() is None:
            no_kinetic_law += 1
            continue
        try:
            reaction = Reaction(
                libsbml_reaction, function_definitions=function_definitions
            )
            k_type = classifyReaction(reaction, function_definitions, species_set)
        except Exception as e:
            print(
                f"Warning: could not classify reaction {reaction_id}: {e}",
                file=sys.stderr,
            )
            k_type = NA
        counts[k_type] += 1

    num_reactions = model.getNumReactions()
    return {
        "species_count": len(species_list),
        "reaction_count": num_reactions,
        "reactions_with_kinetic_law": num_reactions - no_kinetic_law,
        "reactions_without_kinetic_law": no_kinetic_law,
        "rate_law_counts": counts,
    }


def getModelNumsInRange(first_model_num: int, last_model_num: int) -> list[int]:
    """Return sorted model numbers in cn.BIOMODELS_DIR within the inclusive range."""
    model_nums = []
    for entry in os.listdir(cn.BIOMODELS_DIR):
        if not "BIOMD" in entry:
            continue
        model_num = BiomodelsIterator.extractModelNum(entry)
        if first_model_num <= model_num <= last_model_num:
            model_nums.append(model_num)
    return sorted(model_nums)


def getModelName(model_num: int) -> str:
    """Return the BioModel name (e.g. 'BIOMD0000000001') for a model number."""
    return f"BIOMD{model_num:010d}"


def getModelPath(model_num: int) -> str:
    """Return the expected SBML path for a model number in cn.BIOMODELS_DIR."""
    model_name = getModelName(model_num)
    return os.path.join(cn.BIOMODELS_DIR, model_name, f"{model_name}_url.xml")


def makeRow(model_num: int, stats: dict) -> dict:
    """Build one CSV row from a model's rate-law statistics."""
    row = {
        cn.COL_MODEL_NUM: model_num,
        cn.COL_MODEL_NAME: getModelName(model_num),
        cn.COL_NUM_SPECIES: stats["species_count"],
        cn.COL_NUM_REACTION: stats["reaction_count"],
        "num_reaction_with_kinetic_law": stats["reactions_with_kinetic_law"],
        "num_reaction_without_kinetic_law": stats["reactions_without_kinetic_law"],
    }
    row.update(stats["rate_law_counts"])
    return row


def main() -> None:
    """Parse CLI arguments, classify the selected models, and write a CSV."""
    parser = argparse.ArgumentParser(
        description="Classify reaction rate laws for BioModels SBML models "
                    "(SBMLKinetics 2DK scheme)."
    )
    parser.add_argument(
        "--first_model_num",
        type=int,
        default=0,
        help="First BioModel number to process (inclusive).",
    )
    parser.add_argument(
        "--last_model_num",
        type=int,
        default=int(1e9),
        help="Last BioModel number to process (inclusive).",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_PATH,
        help="Path for the output CSV file (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.first_model_num > args.last_model_num:
        print(
            f"Error: first_model_num ({args.first_model_num}) must be "
            f"less than or equal to last_model_num ({args.last_model_num}).",
            file=sys.stderr,
        )
        sys.exit(1)

    model_nums = getModelNumsInRange(args.first_model_num, args.last_model_num)
    # Initialize rows
    rows = []
    if os.path.exists(args.output):
        df = pd.read_csv(args.output)
        rows = [r.to_dict() for i, r in df.iterrows()]
    # Iterate across rows
    for model_num in model_nums:
        sbml_path = getModelPath(model_num)
        if not os.path.exists(sbml_path):
            print(
                f"Invalid model number: {model_num} "
                f"(no SBML file at {sbml_path})",
                file=sys.stderr,
            )
            continue
        model_name = getModelName(model_num)
        print(f"Processing model: {model_name}")
        stats = analyzeModel(sbml_path)
        rows.append(makeRow(model_num, stats))
        # Write on each iteration
        df = pd.DataFrame(rows, columns=CSV_COLUMNS)
        df.to_csv(args.output, index=False)

    if not rows:
        print("Error: no valid models to process.", file=sys.stderr)
        sys.exit(1)

    print(f"Results written to: {args.output} ({len(rows)} model(s))")


if __name__ == "__main__":
    main()

