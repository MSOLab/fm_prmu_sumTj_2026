"""uv run log_checker.py ../Outputs_scenarios/20250923T032521_206759/multi_scenario_runner.log -o matches.txt"""

import argparse
import re
from pathlib import Path
from typing import Iterable

DEFAULT_REGEX = re.compile(
    r"Objective Value and Bound Over Time plot saved to Outputs_scenarios/20250923T032521_206759/output_600s/nehedd_IC1020/\d+/results/\d+_progress_plot\.png"
)


def find_files(paths: Iterable[Path]) -> Iterable[Path]:
    for p in paths:
        if p.is_dir():
            yield from p.rglob("*.txt")
            yield from p.rglob("*.log")
        else:
            yield p


def extract_lines(
    input_paths: list[str],
    output_file: str,
    pattern: re.Pattern = DEFAULT_REGEX,
    dedupe: bool = False,
) -> list[str]:
    out_path = Path(output_file)
    matched = []
    for f in find_files([Path(p) for p in input_paths]):
        try:
            with f.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if pattern.search(line):
                        line_stripped = line.rstrip("\n")
                        entry = f"{f}:{line_stripped}"
                        matched.append(entry)
        except Exception:
            # skip unreadable files
            continue

    if dedupe:
        # preserve order while deduping
        seen = set()
        unique = []
        for m in matched:
            if m not in seen:
                seen.add(m)
                unique.append(m)
        matched = unique

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_f:
        for line in matched:
            out_f.write(line + "\n")

    return matched


def extract_ids_from_line(line: str) -> tuple[str, str] | None:
    """
    Extract the two numeric components in paths like:
    .../{folder_num}/results/{file_num}_progress_plot.png
    Returns (folder_num, file_num) or None if not found.
    """
    m = re.search(r"/(\d+)/results/(\d+)_progress_plot\.png", line)
    if m:
        return m.group(1), m.group(2)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract lines containing 'Objective Value and Bound Over Time plot saved to Outputs_scenarios/..._progress_plot.png' from text files."
    )
    parser.add_argument(
        "inputs", nargs="+", help="Input file(s) or directory(ies) to scan."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="extracted_progress_lines.txt",
        help="Output text file to write matching lines (default: extracted_progress_lines.txt).",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Remove duplicate lines while preserving order.",
    )
    args = parser.parse_args()
    matched_lines = extract_lines(
        args.inputs, args.output, pattern=DEFAULT_REGEX, dedupe=args.dedupe
    )

    expected_numbers = list(range(1, 541, 5))

    for line in matched_lines:
        ids = extract_ids_from_line(line)
        if ids:
            print(f"Extracted IDs from line: {ids}")
            assert len(ids) == 2
            assert ids[0] == ids[1]
            try:
                num = int(ids[0])
                expected_numbers.remove(num)
            except ValueError:
                print(f"Non-integer ID found: {ids[0]}")
        else:
            print("No IDs found in line.")

    print(f"Expected numbers remaining: {expected_numbers}")


if __name__ == "__main__":
    main()
