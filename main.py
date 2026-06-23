import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "sources"))

import airbus_collector
import boeing_collector
import esa_collector
import clean
import index_to_chroma


def main():
    print("STEP 1 — Collecting data from all sources")
    airbus_collector.run()
    boeing_collector.run()
    esa_collector.run()

    print("\nSTEP 2 — Cleaning (noise removal) and de-duplicating")
    clean.clean()

    print("\nSTEP 3 — Embedding and indexing the cleaned data into ChromaDB")
    index_to_chroma.index()

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()