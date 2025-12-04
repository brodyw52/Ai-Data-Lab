import os
import sys
import matplotlib.pyplot as plt
from viz_common import load_model, get_model_features

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    try:
        booster = load_model(HERE)
    except Exception as e:
        print(f"Could not load model: {e}")
        sys.exit(1)

    try:
        importance = booster.get_score(importance_type="gain")
    except Exception as e:
        print(f"Could not get feature importance (gain): {e}")
        sys.exit(1)

    if not importance:
        print("Model returned no feature importance for 'gain'. Trying 'weight' instead.")
        importance = booster.get_score(importance_type="weight")

    if not importance:
        print("No feature importance available from model.")
        sys.exit(1)

    # importance is a dict {feature_name: value}
    items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    keys = [k for k, v in items]
    vals = [v for k, v in items]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(keys)), vals)
    plt.xticks(range(len(keys)), keys, rotation=45, ha='right')
    plt.ylabel('Gain')
    plt.title('XGBoost feature importance (gain)')
    plt.tight_layout()
    out = os.path.join(HERE, 'feature_importance_gain.png')
    plt.savefig(out, dpi=150)
    plt.show()
    print('Saved feature importance plot to', out)

if __name__ == '__main__':
    main()
