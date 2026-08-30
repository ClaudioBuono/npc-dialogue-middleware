import pandas as pd
from core.paths import resource_path
HURTLEX_PATH = resource_path("hurtlex_EN.tsv")


def load_derogatory_terms() -> set[str]:
    """Loads and filters derogatory terms from the HurtLex dataset.

    Reads the HurtLex English TSV file, filters the terms to keep only those
    classified under the 'conservative' confidence level, and extracts a unique
    set of terms.

    TODO: change loaded dataset based on language in settings

    Returns:
        set[str]: A set of unique lemma strings marked as conservative derogatory terms.
    """
    hurtlex_df = pd.read_csv(HURTLEX_PATH, sep="\t")
    return set(hurtlex_df["lemma"].dropna().tolist())