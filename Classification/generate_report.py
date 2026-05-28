from pathlib import Path

from Patrick import generate_report as report

RESULTS_DIR = Path(__file__).parent / "results"


def main(results_dir=RESULTS_DIR):
    report.RESULTS_DIR = Path(results_dir)
    report.main()


if __name__ == "__main__":
    main()
