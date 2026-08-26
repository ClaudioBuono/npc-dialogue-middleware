import pandas as pd


def load_derogatory_terms() -> set[str]:
    """Loads and filters derogatory terms from the HurtLex dataset. 
    
    Reads the HurtLex English TSV file, filters the terms to keep only those
    classified under the 'conservative' confidence level, and extracts a unique
    set of terms.

    TODO: change loaded dataset based on language in settings

    Returns:
        set[str]: A set of unique lemma strings marked as conservative derogatory terms.
    """
    # Load HurtLex TSV file
    hurtlex_df = pd.read_csv("src/tools/hurtlex_EN.tsv", sep="\t")
    
    # Extract unique lemmas, dropping any null values
    return set(hurtlex_df["lemma"].dropna().tolist())