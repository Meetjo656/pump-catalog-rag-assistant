import pandas as pd

def read_csv(path):
    """
    Generic CSV reader.
    Returns list of dicts (one per row).
    """
    df = pd.read_csv(path)
    df = df.fillna("")  # avoid NaN issues
    return df.to_dict(orient="records") 