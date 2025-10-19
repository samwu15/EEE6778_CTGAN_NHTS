
import pandas as pd

def apply_cleaning_rules(df_household: pd.DataFrame, df_person: pd.DataFrame) -> pd.DataFrame:
    """Apply NHTS rules (2)(4)(5) and return a cleaned household dataframe.

    Rules:
      (2) Keep households where HHSIZE equals the number of persons recorded.
      (4) Remove cases with FRSTHM17 marked as whole-house interview (if present).
      (5) Keep only households with all members in-town (OUTOFTWN=0 & OUTCNTRY=0).
    """
    # (2) HHSIZE check using person counts
    person_counts = df_person.groupby('HOUSEID').size().rename('PERSON_COUNT')
    df = df_household.merge(person_counts, on='HOUSEID', how='left')
    df = df[df['HHSIZE'] == df['PERSON_COUNT']]

    # (4) Remove "whole-house interview" if the column exists
    if 'FRSTHM17' in df.columns:
        df = df[df['FRSTHM17'] != 'whole-house']

    # (5) Keep only households where all members are in-town (if columns exist in person file)
    if set(['OUTOFTWN', 'OUTCNTRY']).issubset(df_person.columns):
        in_town = df_person.groupby('HOUSEID').apply(
            lambda g: ((g['OUTOFTWN'] == 0) & (g['OUTCNTRY'] == 0)).all()
        )
        keep_ids = set(in_town[in_town].index)
        df = df[df['HOUSEID'].isin(keep_ids)]

    return df

def basic_features(df: pd.DataFrame) -> pd.DataFrame:
    # Minimal selection; adapt to your columns
    cols = ['HHVEHCNT','HHSIZE','HHFAMINC','URBRUR','CNTTDHH','R_AGE_IMP','DRIVER','WORKER']
    keep = [c for c in cols if c in df.columns]
    return df[keep].copy()
