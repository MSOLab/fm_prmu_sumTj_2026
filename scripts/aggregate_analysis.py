"""Cross-run analysis aggregator.

Gather N scenario directories (from any number of parent run dirs) into a
timestamped ``analysis/<ts>_<label>/`` sub-directory and runs the existing
dashboard pipeline to produce an aggregated report.

Usage:

    uv run python scripts/aggregate_analysis.py \\
        --scenario Outputs_scenarios/<run_ts>/<scenario_a> \\
        --scenario Outputs_scenarios/<run_ts>/<scenario_b> \\
        ...
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import yaml


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


logger = logging.getLogger("aggregate_analysis")


def _resolve_and_validate(path: str) -> Path:
    p = Path(path).resolve()
    if not p.is_dir():
        raise SystemExit(f"ERROR: scenario directory not found: {p}")
    if not (p / "multi_instance_summary.csv").exists():
        raise SystemExit(
            f"ERROR: {p} does not contain multi_instance_summary.csv"
        )
    return p


def _check_basename_uniqueness(paths: list[Path]) -> None:
    seen: dict[str, list[Path]] = {}
    for p in paths:
        seen.setdefault(p.name, []).append(p)
    collisions = {name: plist for name, plist in seen.items() if len(plist) > 1}
    if collisions:
        msg_parts = ["ERROR: duplicate scenario basenames found:"]
        for name, plist in collisions.items():
            for p in plist:
                msg_parts.append(f"  {name} -> {p}")
        msg_parts.append("Use --label or rename/exclude one of them.")
        raise SystemExit("\n".join(msg_parts))


def _default_label(scenario_paths: list[Path]) -> str:
    parts = [p.name for p in scenario_paths]
    joined = "_".join(parts)
    return joined[:40]


def _materialize_scenarios(
    scenario_paths: list[Path],
    analysis_dir: Path,
    mode: str,
) -> None:
    for src in scenario_paths:
        dst = analysis_dir / src.name
        if mode == "symlink":
            dst.mkdir(parents=False, exist_ok=False)
            for item in src.iterdir():
                item_dst = dst / item.name
                if item.is_dir():
                    item_dst.symlink_to(item.resolve(), target_is_directory=True)
                elif item.is_file():
                    item_dst.symlink_to(item.resolve())
        else:
            shutil.copytree(src, dst, symlinks=False, dirs_exist_ok=False)


def _find_metadata_yaml(scenario_dir: Path) -> Path | None:
    parent = scenario_dir.parent
    candidates = sorted(parent.glob("metadata_*.yaml"))
    if not candidates:
        return None
    if len(candidates) > 1:
        logger.warning(
            "Multiple metadata YAMLs in %s; picking %s",
            parent,
            candidates[0].name,
        )
    return candidates[0]


def _resolve_baseline(
    explicit_baseline: Path | None,
    scenario_paths: list[Path],
    repo_root: Path,
) -> tuple[Path | None, dict[str, str]]:
    if explicit_baseline is not None:
        bp = explicit_baseline
        if not bp.is_absolute():
            bp = (repo_root / bp).resolve()
        if bp.exists():
            return bp, {}
        logger.warning("Explicit baseline not found at %s; proceeding without", bp)
        return None, {}

    candidates: set[str] = set()
    mapping: dict[str, str] = {}
    for sp in scenario_paths:
        meta_path = _find_metadata_yaml(sp)
        if meta_path is None:
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f) or {}
        if isinstance(meta, dict) and meta.get("baseline_csv_path"):
            candidates.add(str(meta["baseline_csv_path"]))
            raw_mapping = meta.get("baseline_column_mapping") or {}
            if isinstance(raw_mapping, dict):
                mapping = {
                    "instance": raw_mapping.get("instance", "Instance"),
                    "obj_val": raw_mapping.get("obj_val", "BKS"),
                    "obj_bound": raw_mapping.get("obj_bound", "LB"),
                }

    if len(candidates) == 1:
        bp = Path(next(iter(candidates)))
        if not bp.is_absolute():
            bp = (repo_root / bp).resolve()
        if bp.exists():
            return bp, mapping
        logger.warning("Baseline from metadata not found at %s; proceeding without", bp)
        return None, {}

    if len(candidates) > 1:
        logger.error(
            "Scenarios disagree on baseline_csv_path: %s. "
            "Pass --baseline explicitly.",
            sorted(candidates),
        )
        raise SystemExit(1)

    return None, {}


def _build_info_df_from_disk(scenario_paths: list[Path]) -> list[dict[str, Any]]:
    from output_filenames import OutputFilenames

    records: list[dict[str, Any]] = []
    for sp in scenario_paths:
        flow_path = sp / OutputFilenames.SUBROUTINE_FLOW_CACHE_FN
        stop_path = sp / OutputFilenames.STOPPING_CRITERIA_CACHE_FN
        subroutine_flow = ""
        stopping_criteria = ""
        if flow_path.exists():
            with open(flow_path, "r", encoding="utf-8") as f:
                subroutine_flow = f.read()
        if stop_path.exists():
            with open(stop_path, "r", encoding="utf-8") as f:
                stopping_criteria = f.read()

        description = ""
        meta_path = _find_metadata_yaml(sp)
        if meta_path is not None:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = yaml.safe_load(f) or {}
            dicts = meta.get("dicts_of_i_o_data_path") or []
            if isinstance(dicts, list):
                for entry in dicts:
                    if isinstance(entry, dict) and entry.get("output_dir") == sp.name:
                        description = entry.get("description", "")
                        break

        records.append(
            {
                "Scenario": sp.name,
                "Subroutine Flow": subroutine_flow,
                "Stopping Criteria": stopping_criteria,
                "Description": description,
            }
        )
    return records


def _write_manifest(
    analysis_dir: Path,
    scenario_paths: list[Path],
    baseline_path: Path | None,
    label: str | None,
    ts: str,
    link_mode: str,
    cli_argv: list[str],
) -> None:
    manifest = {
        "timestamp": ts,
        "label": label or "",
        "link_mode": link_mode,
        "scenarios": [str(p.resolve()) for p in scenario_paths],
        "baseline_csv_path": str(baseline_path.resolve()) if baseline_path else None,
        "cli_argv": cli_argv,
    }
    manifest_path = analysis_dir / "analysis_manifest.yaml"
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
    logger.info("Wrote manifest to %s", manifest_path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        action="append",
        required=True,
        dest="scenarios",
        help="Path to a scenario directory (repeatable)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Path to baseline CSV (absolute or repo-relative)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Human-readable slug appended to output dir name",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("analysis"),
        help="Root output directory (default: analysis/)",
    )
    parser.add_argument(
        "--link",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to materialize scenario dirs (default: symlink)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.log_level)

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    scenario_paths = [_resolve_and_validate(s) for s in args.scenarios]
    _check_basename_uniqueness(scenario_paths)

    ts = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    label = args.label or _default_label(scenario_paths)
    out_root = args.out_root
    if not out_root.is_absolute():
        out_root = (repo_root / out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    analysis_dir = out_root / f"{ts}_{label}"
    if analysis_dir.exists():
        logger.warning("%s already exists; regenerating timestamp", analysis_dir)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
        analysis_dir = out_root / f"{ts}_{label}"
    analysis_dir.mkdir(parents=True, exist_ok=False)
    logger.info("Analysis dir: %s", analysis_dir)

    _materialize_scenarios(scenario_paths, analysis_dir, args.link)

    from flowshop_tardiness.report.dashboards import (  # noqa: E402
        write_post_run_dashboard_artifacts,
    )
    from flowshop_tardiness.report.dashboards.multi_scenario_report import (  # noqa: E402
        DEFAULT_RPD_FORMATS,
        DEFAULT_STAT_PAIRS,
        aggregate_scenario_summaries,
        build_dashboard_df,
        build_info_df,
        write_multi_scenario_excel_report,
    )

    summary_df = aggregate_scenario_summaries(
        scenario_paths, out_dir=analysis_dir
    )

    baseline_path, column_mapping = _resolve_baseline(
        args.baseline, scenario_paths, repo_root
    )
    baseline_df = None
    baseline_instance_col = column_mapping.get("instance", "Instance")
    baseline_obj_val_col = column_mapping.get("obj_val", "BKS")
    baseline_obj_bound_col = column_mapping.get("obj_bound", "LB")
    if baseline_path is not None:
        import pandas as pd

        baseline_df = pd.read_csv(baseline_path)
        logger.info("Loaded baseline from %s", baseline_path)

    info_records = _build_info_df_from_disk(scenario_paths)
    info_df = build_info_df(info_records)

    dashboard_df = build_dashboard_df(
        summary_df,
        baseline_df=baseline_df,
        baseline_instance_col=baseline_instance_col,
        baseline_obj_val_col=baseline_obj_val_col,
        baseline_obj_bound_col=baseline_obj_bound_col,
        base_output_metadata=None,
        stat_pairs=DEFAULT_STAT_PAIRS,
        rpd_col_formats=DEFAULT_RPD_FORMATS,
    )

    write_multi_scenario_excel_report(
        analysis_dir / "multi_scenario_report.xlsx",
        dashboard_df=dashboard_df,
        raw_summary_df=summary_df,
        info_df=info_df,
        baseline_df=baseline_df,
    )

    written = write_post_run_dashboard_artifacts(
        analysis_dir,
        summary_csv=analysis_dir / "all_scenarios_summary.csv",
        baseline_df=baseline_df,
        baseline_instance_col=baseline_instance_col,
        baseline_obj_val_col=baseline_obj_val_col,
        baseline_obj_bound_col=baseline_obj_bound_col,
        run_id=ts,
        scenario_output_root=analysis_dir,
    )

    _write_manifest(
        analysis_dir,
        scenario_paths,
        baseline_path,
        args.label,
        ts,
        args.link,
        list(sys.argv) if argv is None else list(argv),
    )

    logger.info("Wrote %d artifact(s):", len(written))
    for key, path in written.items():
        logger.info("  [%s] %s", key, path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
