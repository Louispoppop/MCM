import pandas as pd

INPUT_PATH = "Cleaned_data_with_voting_rate.csv"
OUTPUT_PATH = "Cleaned_data_with_votes.csv"
TOTAL_VOTES = 14_000_000


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    df["votes"] = (df["voting_rate"] * TOTAL_VOTES).round().astype("Int64")

    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
