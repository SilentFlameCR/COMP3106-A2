import sys
from assignment2 import naive_bayes_classifier

# Default snake_measurements per example index
EX_MEASUREMENTS = {
    0: [350, 42, 13],
    1: [390, 28, 13],
    2: [340, 26, 12],
    3: [350, 42, 13],
}

def run_example(idx: int, measurements=None):
    if measurements is None:
        measurements = EX_MEASUREMENTS[idx]
    dataset_path = f"./Examples/Example{idx}/dataset.csv"
    most_class, probs = naive_bayes_classifier(dataset_path, measurements)
    probs_str = "[" + ", ".join(f"{p}" for p in probs) + "]"
    print(f"Example {idx} | measurements={measurements} -> "
          f"probs={probs_str} | most likely: {most_class}")

def main():
    args = sys.argv[1:]
    if not args:
        # run all 0..3 with default measurements
        for i in range(4):
            run_example(i)
        return

    # 1 arg: example index with default measurements
    if len(args) == 1:
        idx = int(args[0])
        if idx not in EX_MEASUREMENTS:
            raise SystemExit("Example index must be one of 0, 1, 2, 3.")
        run_example(idx)
        return

    # 4 args: example index + L W S
    if len(args) == 4:
        idx = int(args[0])
        if idx not in EX_MEASUREMENTS:
            raise SystemExit("Example index must be one of 0, 1, 2, 3.")
        L = float(args[1]); W = float(args[2]); S = float(args[3])
        run_example(idx, [L, W, S])
        return

    raise SystemExit("Usage:\n"
                     "  python test_nb.py\n"
                     "  python test_nb.py <idx>\n"
                     "  python test_nb.py <idx> <length> <weight> <speed>")

if __name__ == "__main__":
    main()