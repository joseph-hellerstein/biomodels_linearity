"""
Classify reaction rate laws for BioModels SBML models using the SBMLKinetics 2DK scheme.

Iterates over BioModels models in cn.BIOMODELS_DIR within a model-number range,
classifies each reaction's kinetic law into one of twelve mutually exclusive
kinetics types (ZERO, UNDR1, UNDR2, UNMO, BIDR1, BIDR2, BIMO, MM, MMCAT,
HILL, FR, NA), and writes the per-model counts and reaction-name lists to a
CSV file.

The classification system is from:

@article{xu_sbmlkinetics_2023,
	title = {{SBMLKinetics}: a tool for annotation-independent classification of reaction
    kinetics for {SBML} models},
	volume = {24},
	issn = {1471-2105},
	url = {https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-023-05380-3},
	doi = {10.1186/s12859-023-05380-3},

Zeroth order (ZERO) the number of species in the kinetic law was zero; 
Uni-directional mass action (UNDR) the kinetic law was a single product of terms
    and all species in the kinetic law were reactants. UNDR is split by total
    reactant stoichiometry (stoichiometric coefficients are summed): UNDR1 has
    exactly 1 and UNDR2 has any other total.
Uni-term with the moderator (UNMO) the kinetic law was a single product of terms
    and not UNDR, namely at least one species in the kinetic law was not a reactant
Bi-directional mass action (BIDR) the kinetic law was the difference between two  products of terms,
    and the first product of terms contained species as all the reactants
    while the second product of terms contained species as all the products.
    BIDR is split by total stoichiometry: BIDR1 has exactly 1 reactant and
    exactly 1 product, while BIDR2 has any other total.
Bi-terms with the moderator (BIMO) the kinetic law was the difference between two products of terms
    and NOT BIDR
Michaelis–Menten kinetics without explicit enzyme (MM) the kinetic law was in the
    format of Michaelis–Menten expressions without an explicit enzyme
Michaelis–Menten kinetics with an explicit enzyme (MMCAT) the kinetic law was in
    the format of Michaelis–Menten expressions with an explicit enzyme
Hill equation (HILL) the kinetic law was in the format of Hill equations
Fraction format other than MM, MMCAT, and HILL (FR) the kinetic law was in
    the format of fraction with at least one species in the denominator and NOT MM, MMCAT, or HILL
Not classified (NA) not classified kinetics.

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
from src.timecourse_iterator import TimecourseIterator  # type: ignore

import argparse
import math
import os
import sys

import pandas as pd  # type: ignore
from sympy import im, simplify  # type: ignore

import libsbml  # type: ignore
from SBMLKinetics.common.function_definition import FunctionDefinition  # type: ignore
from SBMLKinetics.common.reaction import Reaction  # type: ignore

# Excluded models
EXCLUDED_MODELS = [64, 849]
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
# Output sub-labels that replace UNDR/BIDR, split by total stoichiometry.
K_TYPE_SUBDIVISIONS = {
    "UNDR": ["UNDR1", "UNDR2"],
    "BIDR": ["BIDR1", "BIDR2"],
}
K_TYPE_LABELS = [
    sub for label, _ in K_TYPE_CHECKS
    for sub in K_TYPE_SUBDIVISIONS.get(label, [label])
] + [NA]


def getReactionNameColumn(label: str) -> str:
    """Return the CSV column name holding a label's reaction-name list."""
    return f"{label}_reaction_names"


REACTION_NAME_COLUMNS = [getReactionNameColumn(label) for label in K_TYPE_LABELS]
OUTPUT_FILENAME = "rate_law_statistics.csv"
OUTPUT_PATH = os.path.join(cn.DATA_DIR, OUTPUT_FILENAME)
CSV_COLUMNS = [cn.COL_MODEL_NUM, cn.COL_SYSTEM_ID, cn.COL_NUM_SPECIES,
        cn.COL_NUM_REACTION, "num_reaction_with_kinetic_law",
        "num_reaction_without_kinetic_law"] + K_TYPE_LABELS \
    + REACTION_NAME_COLUMNS


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


def totalStoichiometry(species_refs: list[libsbml.SpeciesReference]) -> float:
    """Sum the stoichiometries of a list of species references.

    References without a set stoichiometry (NaN) count as 1, per the SBML
    default. An empty list sums to 0.
    """
    total = 0.0
    for ref in species_refs:
        stoich = float(ref.getStoichiometry())
        total += 1.0 if math.isnan(stoich) else stoich
    return total


def subdivideLabel(label: str, num_reactants: float, num_products: float) -> str:
    """Split UNDR/BIDR into subtypes by total reactant/product stoichiometry.

    UNDR1 has exactly 1 reactant and BIDR1 has exactly 1 reactant and
    exactly 1 product; any other total yields the "2" subtype. Other labels
    are returned unchanged.
    """
    if label == "UNDR":
        return "UNDR1" if num_reactants == 1 else "UNDR2"
    if label == "BIDR":
        return ("BIDR1" if num_reactants == 1 and num_products == 1
                else "BIDR2")
    return label


def classifyReaction(
    reaction: Reaction,
    function_definitions: list[FunctionDefinition],
    species_set: set[str],
) -> str:
    """Classify one reaction's kinetic law and return a K-type label.

    Runs the SBMLKinetics type checks in order and returns the first match,
    or NA if none apply. A check that raises on unusual kinetics is treated
    as a non-match so one odd reaction cannot abort the whole model. UNDR and
    BIDR matches are subdivided into "1"/"2" subtypes by total reactant and
    product stoichiometry.
    """
    reaction.kinetic_law.mkSymbolExpression(function_definitions)

    reactant_list = [r.getSpecies() for r in reaction.reactants]
    product_list = [p.getSpecies() for p in reaction.products]
    num_reactants = totalStoichiometry(reaction.reactants)
    num_products = totalStoichiometry(reaction.products)

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
                return subdivideLabel(label, num_reactants, num_products)
        except Exception:
            continue
    return NA


def analyzeModel(sbmlPath: str) -> dict:
    """Parse sbmlPath and return rate-law statistics for the model.

    Returns per-type reaction counts plus, for each type, the list of
    reaction ids (in model order) classified as that type.
    """
    model = readModel(sbmlPath)
    function_definitions = getFunctionDefinitions(model)
    species_list = [model.getSpecies(n).getId() for n in range(model.getNumSpecies())]
    species_set = set(species_list)

    counts = {label: 0 for label in K_TYPE_LABELS}
    rate_law_reactions : dict  = {label: [] for label in K_TYPE_LABELS}
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
        rate_law_reactions[k_type].append(reaction_id)

    num_reactions = model.getNumReactions()
    return {
        "species_count": len(species_list),
        "reaction_count": num_reactions,
        "reactions_with_kinetic_law": num_reactions - no_kinetic_law,
        "reactions_without_kinetic_law": no_kinetic_law,
        "rate_law_counts": counts,
        "rate_law_reactions": rate_law_reactions,
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
        cn.COL_SYSTEM_ID: getModelName(model_num),
        cn.COL_NUM_SPECIES: stats["species_count"],
        cn.COL_NUM_REACTION: stats["reaction_count"],
        "num_reaction_with_kinetic_law": stats["reactions_with_kinetic_law"],
        "num_reaction_without_kinetic_law": stats["reactions_without_kinetic_law"],
    }
    row.update(stats["rate_law_counts"])
    for label in K_TYPE_LABELS:
        row[getReactionNameColumn(label)] = ",".join(
            stats["rate_law_reactions"][label])
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

    selected_model_nums = TimecourseIterator.getModelNums()
    # Initialize rows
    rows = []
    if os.path.exists(args.output):
        df = pd.read_csv(args.output)
        old_columns = [c for c in ("UNDR", "BIDR") if c in df.columns]
        if old_columns:
            print(
                f"Warning: {args.output} has pre-subdivision columns "
                f"{old_columns}; its rows will have NaN in the new "
                f"UNDR1/UNDR2/BIDR1/BIDR2 columns. Delete the file to "
                f"reclassify all models.",
                file=sys.stderr,
            )
        if not set(REACTION_NAME_COLUMNS) <= set(df.columns):
            print(
                f"Warning: {args.output} predates the reaction-name "
                f"columns; its rows will have empty *_reaction_names "
                f"cells. Delete the file to reclassify all models.",
                file=sys.stderr,
            )
        rows = [r.to_dict() for i, r in df.iterrows()]
    # Iterate across rows
    processed_model_nums = [r[cn.COL_MODEL_NUM] for r in rows] + EXCLUDED_MODELS
    #for model_num in model_nums:
    for item in BiomodelsIterator(first_model_num=args.first_model_num,
            last_model_num=args.last_model_num):
        model_num = item.model_num
        if not model_num in selected_model_nums:
            continue
        if model_num in processed_model_nums:
            continue
        sbml_path = item.sbml_paths[0]
        if not os.path.exists(sbml_path):
            print(
                f"Invalid model number: {model_num} "
                f"(no SBML file at {sbml_path})",
                file=sys.stderr,
            )
            continue
        stats = analyzeModel(sbml_path)
        rows.append(makeRow(model_num, stats))
        # Write on each iteration
        df = pd.DataFrame(rows, columns=CSV_COLUMNS)
        df.to_csv(args.output, index=False)
        processed_model_nums.append(model_num)

    if not rows:
        print("Error: no valid models to process.", file=sys.stderr)
        sys.exit(1)

    print(f"Results written to: {args.output} ({len(rows)} model(s))")


if __name__ == "__main__":
    main()

