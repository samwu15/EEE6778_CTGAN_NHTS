
import argparse
import pandas as pd
from sdv.tabular import CTGAN

def main(args):
    df = pd.read_csv(args.data)
    # Identify categorical columns (example)
    categorical = [c for c in df.columns if df[c].dtype == 'object']
    model = CTGAN(epochs= args.epochs, batch_size=args.batch_size, verbose=True)
    model.fit(df)
    model.save(args.out.replace('.pkl','.ctgan'))
    synth = model.sample(n_rows=min(len(df), args.n_samples))
    synth.to_csv(args.out.replace('.pkl','.csv'), index=False)
    print(f"Saved model -> {args.out.replace('.pkl','.ctgan')}\nSaved samples -> {args.out.replace('.pkl','.csv')}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True, help='CSV path for training')
    p.add_argument('--out', type=str, required=True, help='Output path prefix (model/checkpoints)')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_samples', type=int, default=1000)
    args = p.parse_args()
    main(args)
