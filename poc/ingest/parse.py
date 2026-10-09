"""Build db/pages.parquet for the active dataset (POC_DATASET)."""
from poc import data


def main() -> None:
    data.build_pages()


if __name__ == "__main__":
    main()
