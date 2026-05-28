from pathlib import Path

from Patrick import baselines

RESULTS_DIR = Path(__file__).parent / "results"


def main(results_dir=RESULTS_DIR):
    results_dir = Path(results_dir)
    baselines.RESULTS_DIR = results_dir
    baselines.generate_report.RESULTS_DIR = results_dir
    baselines.main()


if __name__ == "__main__":
    main()
