import argparse

from src.pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="CUAD clause extraction & summarization pipeline")
    parser.add_argument("--pdf_dir", required=True, help="Path to folder of CUAD contract PDFs")
    parser.add_argument("--output", default="outputs/results.csv", help="Output CSV or JSON path")
    parser.add_argument("--n_contracts", type=int, default=50, help="How many contracts to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--model", default=None, help="Override Gemini model name")
    parser.add_argument(
        "--reset_output",
        action="store_true",
        help="Delete the existing output file before starting a fresh run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results, stats = run_pipeline(
        pdf_dir=args.pdf_dir,
        output_path=args.output,
        n_contracts=args.n_contracts,
        seed=args.seed,
        model=args.model,
        reset_output=args.reset_output,
    )

    skipped_count = stats.get("skipped_count", 0)
    print(
        f"Done. {stats['successful_count']}/{stats['processed_count']} contracts processed successfully in this run."
    )
    print(f"Skipped {skipped_count} already completed rows; output now contains {stats['rows_in_output']} rows in {args.output}")
    if stats.get("stopped_early"):
        print(f"Stopped early due to {stats['stop_reason']}; partial results were saved to {args.output}.")


if __name__ == "__main__":
    main()
