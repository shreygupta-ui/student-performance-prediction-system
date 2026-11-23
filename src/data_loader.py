"""Loading and cleaning the UCI Student Performance (Mathematics) dataset."""
import pandas as pd
from .config import DATA_PATH, EXCLUDE_ZERO_GRADES, FEATURES, TARGET


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    """Read the CSV. The UCI file uses ';' as the separator."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download student-mat.csv from "
            "https://archive.ics.uci.edu/dataset/320/student+performance "
            "and place it in the 'data' folder (see README).")
    return pd.read_csv(path, sep=";")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: strip text, drop duplicates, validate ranges."""
    df = df.copy()
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].str.strip()
    df = df.drop_duplicates().reset_index(drop=True)
    # Grades must lie on the 0-20 scale
    for g in ["G1", "G2", "G3"]:
        df = df[df[g].between(0, 20)]
    return df.reset_index(drop=True)


def load_clean() -> pd.DataFrame:
    return clean(load_raw())


def modeling_data(df: pd.DataFrame) -> pd.DataFrame:
    """Rows used for model training (drops zero-grade non-completers if configured)."""
    return df[df[TARGET] > 0].reset_index(drop=True) if EXCLUDE_ZERO_GRADES else df


def get_xy(df: pd.DataFrame, features=None):
    """Split into feature matrix X and target y (G3)."""
    features = features or FEATURES
    return df[features], df[TARGET]


def dataset_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing": int(df.isna().sum().sum()),
        "mean_g3": float(df[TARGET].mean()),
        "pass_rate": float((df[TARGET] >= 10).mean() * 100),
        "zero_g3": int((df[TARGET] == 0).sum()),
    }
