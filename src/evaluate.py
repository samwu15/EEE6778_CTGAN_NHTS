
import argparse, json
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

def kolmogorov_smirnov(real: pd.Series, synth: pd.Series):
    # numeric only
    r = pd.to_numeric(real, errors='coerce').dropna()
    s = pd.to_numeric(synth, errors='coerce').dropna()
    if len(r) > 2 and len(s) > 2:
        stat, p = ks_2samp(r, s)
        return {'ks_stat': float(stat), 'p_value': float(p)}
    return {'ks_stat': None, 'p_value': None}

def correlation_distance(real_df: pd.DataFrame, synth_df: pd.DataFrame):
    rc = real_df.corr(numeric_only=True).fillna(0.0)
    sc = synth_df.corr(numeric_only=True).fillna(0.0)
    diff = np.abs(rc - sc).values
    return float(diff.mean())

def main(args):
    real = pd.read_csv(args.real)
    synth = pd.read_csv(args.synthetic)
    report = {'per_feature_ks': {}, 'mean_corr_distance': None}

    commons = [c for c in real.columns if c in synth.columns]
    for c in commons:
        if pd.api.types.is_numeric_dtype(real[c]):
            report['per_feature_ks'][c] = kolmogorov_smirnov(real[c], synth[c])

    report['mean_corr_distance'] = correlation_distance(real[commons], synth[commons])

    with open(args.report, 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--real', required=True, help='Path to real CSV')
    p.add_argument('--synthetic', required=True, help='Path to synthetic CSV')
    p.add_argument('--report', required=True, help='Output JSON path')
    args = p.parse_args()
    main(args)
