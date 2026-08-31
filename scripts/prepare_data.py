from __future__ import annotations

import json

from return_risk.config import PROCESSED_DIR, RAW_DATA_PATH
from return_risk.data import load_raw_data
from return_risk.splits import chronological_split, write_splits


def main() -> None:
    frame = load_raw_data(RAW_DATA_PATH)
    splits = chronological_split(frame)
    manifest = write_splits(splits, PROCESSED_DIR, RAW_DATA_PATH)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

