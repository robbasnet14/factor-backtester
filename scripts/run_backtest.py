"""Entry point: wire everything together. Grows as you complete each step."""
import argparse
from src.utils.config import load_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    print("Loaded config:", cfg["universe"]["name"], cfg["universe"]["start_date"], "->", cfg["universe"]["end_date"])
    # TODO Step 2: data = load ...
    # TODO Step 3: factors = compute ...
    # TODO Step 4-5: weights -> backtest with costs
    # TODO Step 6: metrics + plots

if __name__ == "__main__":
    main()
