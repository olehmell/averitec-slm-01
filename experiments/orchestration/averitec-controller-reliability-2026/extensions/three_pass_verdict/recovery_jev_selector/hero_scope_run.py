"""Scoped entry point for the original HerO runner: Jev recovery pass 4 only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import verdict_runner as original  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--astra-gate", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.plan_only == args.execute:
        raise ValueError("choose_plan_or_execute")
    if args.plan_only:
        _, _, result = original.prepare(args.manifest, args.candidates, args.packages,
                                        args.tokenizer_dir, arms=("jev",), passes=(4,), expected_cases=100)
        if result["packages"] != 100 or result["ready_packages"] != 100 or result["maximum_actual_calls"] != 101:
            raise ValueError("scoped_plan_not_100_ready")
    else:
        if args.output is None:
            raise ValueError("output_required")
        result = original.execute(args.manifest, args.candidates, args.packages,
                                  args.tokenizer_dir, args.output,
                                  astra_gate=args.astra_gate, endpoint=args.endpoint,
                                  arms=("jev",), passes=(4,), expected_cases=100)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
