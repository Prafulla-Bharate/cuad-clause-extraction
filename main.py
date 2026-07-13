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
        "--batch_start",
        type=int,
        default=0,
        help="Start index within the sampled contract subset for this run",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Number of sampled contracts to process in this run; omit to process the rest",
    )
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
        batch_start=args.batch_start,
        batch_size=args.batch_size,
        reset_output=args.reset_output,
    )

    print(
        f"Done. {stats['successful_count']}/{stats['processed_count']} contracts processed successfully in this run."
    )
    print(f"Output now contains {stats['rows_in_output']} rows in {args.output}")


if __name__ == "__main__":
    main()
