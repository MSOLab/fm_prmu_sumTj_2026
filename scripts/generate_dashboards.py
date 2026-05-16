"""Regenerate the HTML dashboards for an existing run directory.

Usage:

    uv run python scripts/generate_dashboards.py <run_dir> [--baseline <csv>]
                                                          [--metadata <yaml>]

When ``--baseline`` is omitted the script discovers the baseline CSV path
from the saved metadata YAML inside ``<run_dir>`` (any
``metadata_*.yaml`` — picks the unique match). Use ``--metadata`` to point
at a specific file when more than one is present. Run from the repo root.

The script reuses :func:`write_post_run_dashboard_artifacts` so the output
is identical to what ``FsMultiScenarioRunner.post_run_process`` writes at
the end of a normal run — no in-memory state required.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml


def _find_metadata_yaml(run_dir: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.is_absolute() else (run_dir / explicit)
    candidates = sorted(run_dir.glob("metadata_*.yaml"))
    if not candidates:
        return None
    if len(candidates) > 1:
        logging.warning(
            "Multiple metadata YAMLs in %s; picking %s (use --metadata to override)",
            run_dir,
            candidates[0].name,
        )
    return candidates[0]


def _resolve_baseline_from_metadata(
    metadata_path: Path, repo_root: Path
) -> tuple[Path | None, dict]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}
    if not isinstance(meta, dict):
        return None, {}
    baseline_rel = meta.get("baseline_csv_path")
    if not baseline_rel:
        return None, meta
    baseline_path = Path(baseline_rel)
    if not baseline_path.is_absolute():
        baseline_path = repo_root / baseline_path
    return baseline_path, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Path to the run directory")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Path to baseline CSV (defaults to baseline_csv_path from metadata YAML)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Specific metadata YAML inside the run dir (defaults to glob metadata_*.yaml)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repo root for resolving relative baseline paths (default: parent of scripts/)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.is_dir():
        logging.error("Run directory not found: %s", run_dir)
        return 1

    # Make the repo-rooted package imports work when this script is invoked
    # from anywhere (it lives in scripts/, which is not a package).
    repo_root: Path = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from flowshop_tardiness.report.dashboards import (  # noqa: E402  (post sys.path)
        write_post_run_dashboard_artifacts,
    )
    from fs_config import BaselineColumnMapping  # noqa: E402

    baseline_csv: Path | None = args.baseline
    column_mapping = BaselineColumnMapping()

    if baseline_csv is None:
        metadata_path = _find_metadata_yaml(run_dir, args.metadata)
        if metadata_path is None or not metadata_path.exists():
            logging.warning(
                "No metadata YAML found in %s; dashboards will run without baseline",
                run_dir,
            )
        else:
            baseline_csv, meta = _resolve_baseline_from_metadata(
                metadata_path, repo_root
            )
            logging.info("Loaded baseline path from %s", metadata_path)
            mapping = meta.get("baseline_column_mapping") or {}
            if isinstance(mapping, dict) and mapping:
                column_mapping = BaselineColumnMapping(**mapping)

    if baseline_csv is not None:
        baseline_csv = baseline_csv.resolve()
        if not baseline_csv.exists():
            logging.warning(
                "Baseline CSV not found at %s; proceeding without baseline",
                baseline_csv,
            )
            baseline_csv = None

    written = write_post_run_dashboard_artifacts(
        run_dir,
        baseline_csv_path=baseline_csv,
        baseline_instance_col=column_mapping.instance,
        baseline_obj_val_col=column_mapping.obj_val,
        baseline_obj_bound_col=column_mapping.obj_bound,
    )

    if not written:
        logging.warning("No artifacts written for %s", run_dir)
        return 1

    logging.info("Wrote %d artifact(s):", len(written))
    for key, path in written.items():
        logging.info("  [%s] %s", key, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
