"""Build verified native-domain MaxEnt inputs; no fits or final splits."""
import argparse
import json
from pathlib import Path

from wetland_coupling.maxent_inputs import build_formal_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    manifest = build_formal_inputs(args.config,args.out)
    print(json.dumps({"status":manifest["status"],"common_valid_cells":manifest["common_valid_cells"],
                      "seasons":manifest["seasons"]}))


if __name__ == "__main__":
    main()
