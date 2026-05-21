from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import rc_context

from ..graph_model.single_mc_pmtn import PreemptiveBlock, SingleMachinePreemptionMcf


class PreemptiveLbBreakdownPlotter:
    """3-panel breakdown of the preemptive last-stage LB schedule.

    Panel 1: per-job lane Gantt with block penalty labels and tardy hatching.
    Panel 2: per-slot marginal cost (step) timeline.
    Panel 3: per-job total contribution sorted desc.
    """

    cmap_name = "tab20"
    fig_width = 16.0
    height_ratios = (3.0, 1.0, 1.5)
    bar_height = 0.7
    bar_alpha = 0.85
    tardy_hatch = "////"
    grid_alpha = 0.4
    # Scale-adaptive thresholds
    label_min_pct = 0.5  # only annotate blocks contributing >= 0.5% of objective
    label_top_k = 40  # always annotate the K largest blocks even if below pct
    bar_top_k = 30  # show top-K jobs in contribution panel; rest into "others"

    def export(self, file_path: Path, mdl: SingleMachinePreemptionMcf) -> None:
        with rc_context(
            {
                "svg.fonttype": "none",
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial"],
                "text.usetex": False,
            }
        ):
            self._plot(mdl)
            plt.savefig(file_path, bbox_inches="tight", dpi=200)
            logging.info(f"Preemptive LB breakdown saved to {file_path}")
            plt.close()

    def _plot(self, mdl: SingleMachinePreemptionMcf) -> None:
        blocks = mdl.get_blocks()
        objective = mdl.get_obj_value()
        job_to_color = self._make_color_map(mdl.calJ)

        # Height scales with number of job lanes so labels stay readable.
        # Cap so we don't produce a 50-inch figure for huge instances.
        gantt_height = min(max(4.0, 0.18 * len(mdl.calJ)), 30.0)
        bar_panel_height = min(max(2.0, 0.18 * min(len(mdl.calJ), self.bar_top_k + 1)), 8.0)
        total_h = gantt_height + 1.5 + bar_panel_height
        height_ratios = (gantt_height, 1.5, bar_panel_height)

        fig, (ax_gantt, ax_cost, ax_bar) = plt.subplots(
            3,
            1,
            figsize=(self.fig_width, total_h),
            gridspec_kw={"height_ratios": list(height_ratios)},
        )
        title = (
            f"Preemptive LB breakdown — {mdl.name}  |  objective = {objective}  |  "
            f"|J|={len(mdl.calJ)}, T={mdl.t_max}, blocks={len(blocks)}"
        )
        fig.suptitle(title, fontsize=11, y=0.995)

        self._draw_gantt(ax_gantt, mdl, blocks, objective, job_to_color)
        self._draw_cost_timeline(ax_cost, mdl, blocks, job_to_color)
        self._draw_contribution_bar(ax_bar, mdl, blocks, objective, job_to_color)

        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))

    # -----------------------------
    # Panel 1: Job-laned Gantt
    # -----------------------------
    def _draw_gantt(
        self,
        ax,
        mdl: SingleMachinePreemptionMcf,
        blocks: list[PreemptiveBlock],
        objective: int,
        job_to_color: dict[str, tuple[float, float, float, float]],
    ) -> None:
        job_to_y = {j: i for i, j in enumerate(mdl.calJ)}

        # r_j shaded region (0..r_j on each row) — visually marks "release lock"
        for j, y in job_to_y.items():
            rj = mdl.r[j]
            if rj > 0:
                ax.add_patch(
                    patches.Rectangle(
                        (0, y - self.bar_height / 2),
                        rj,
                        self.bar_height,
                        facecolor="lightgray",
                        edgecolor="none",
                        alpha=0.35,
                        zorder=0,
                    )
                )

        # d_j vertical dashed marker per job row
        for j, y in job_to_y.items():
            dj = mdl.d[j]
            ax.plot(
                [dj, dj],
                [y - self.bar_height / 2, y + self.bar_height / 2],
                linestyle="--",
                color="red",
                linewidth=1.0,
                alpha=0.75,
                zorder=2,
            )

        # Decide which blocks get textual annotations.
        # Skip cost-0 blocks (uninformative) and small contributors at scale.
        denom = objective if objective > 0 else 1
        nonzero = [blk for blk in blocks if blk.cost > 0]
        label_pct_cutoff = (self.label_min_pct / 100.0) * denom
        topk_costs = sorted((blk.cost for blk in nonzero), reverse=True)[: self.label_top_k]
        topk_threshold = topk_costs[-1] if topk_costs else 0
        annotate_threshold = min(label_pct_cutoff, topk_threshold) if topk_threshold > 0 else label_pct_cutoff

        # Blocks
        for blk in blocks:
            y = job_to_y[blk.job_id]
            x0 = blk.start_t - 1  # unit slot t covers (t-1, t]
            width = blk.end_t - blk.start_t + 1
            color = job_to_color[blk.job_id]
            ax.add_patch(
                patches.Rectangle(
                    (x0, y - self.bar_height / 2),
                    width,
                    self.bar_height,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=self.bar_alpha,
                    zorder=3,
                )
            )
            # Tardy overlay: slots strictly after d_j
            dj = mdl.d[blk.job_id]
            tardy_left = max(x0, dj)
            tardy_right = x0 + width
            if tardy_right > tardy_left:
                ax.add_patch(
                    patches.Rectangle(
                        (tardy_left, y - self.bar_height / 2),
                        tardy_right - tardy_left,
                        self.bar_height,
                        facecolor="none",
                        edgecolor="black",
                        hatch=self.tardy_hatch,
                        linewidth=0.0,
                        alpha=0.9,
                        zorder=4,
                    )
                )
            if blk.cost > 0 and blk.cost >= annotate_threshold:
                pct = 100.0 * blk.cost / denom
                ax.text(
                    x0 + width / 2,
                    y,
                    f"Σc={blk.cost} ({pct:.0f}%)",
                    ha="center",
                    va="center",
                    fontsize=7,
                    zorder=5,
                )

        ax.set_xlim(0, mdl.t_max + 1)
        ax.set_ylim(-0.6, len(mdl.calJ) - 0.4)
        ax.invert_yaxis()
        # Thin y-ticks if too many jobs to avoid overlapping text.
        n_jobs = len(mdl.calJ)
        max_ticks = 60
        if n_jobs <= max_ticks:
            tick_ys = list(job_to_y.values())
            tick_labels = list(job_to_y.keys())
            tick_fs = 8
        else:
            stride = (n_jobs + max_ticks - 1) // max_ticks
            tick_ys = [y for j, y in job_to_y.items() if y % stride == 0]
            tick_labels = [j for j, y in job_to_y.items() if y % stride == 0]
            tick_fs = 6
        ax.set_yticks(tick_ys)
        ax.set_yticklabels(tick_labels, fontsize=tick_fs)
        ax.set_ylabel("job")
        ax.set_title(
            "Block Gantt (hatched = tardy slots t > d_j; red dashed = d_j; gray = pre-r_j)",
            fontsize=9,
            loc="left",
        )
        ax.grid(True, axis="x", linestyle=":", alpha=self.grid_alpha)

    # -----------------------------
    # Panel 2: Per-slot marginal cost
    # -----------------------------
    def _draw_cost_timeline(
        self,
        ax,
        mdl: SingleMachinePreemptionMcf,
        blocks: list[PreemptiveBlock],
        job_to_color: dict[str, tuple[float, float, float, float]],
    ) -> None:
        # Per block, draw a sequence of unit-width bars colored by job, height = c_{j,t}
        for blk in blocks:
            j = blk.job_id
            color = job_to_color[j]
            for t in range(blk.start_t, blk.end_t + 1):
                c = mdl.c[j][t]
                if c <= 0:
                    continue
                ax.add_patch(
                    patches.Rectangle(
                        (t - 1, 0),
                        1,
                        c,
                        facecolor=color,
                        edgecolor="none",
                        alpha=0.85,
                    )
                )

        # Determine y-limit
        max_c = 0
        for blk in blocks:
            j = blk.job_id
            for t in range(blk.start_t, blk.end_t + 1):
                if mdl.c[j][t] > max_c:
                    max_c = mdl.c[j][t]
        ax.set_xlim(0, mdl.t_max + 1)
        ax.set_ylim(0, max(1, max_c) * 1.1)
        ax.set_xlabel("time t")
        ax.set_ylabel("c_{j(t),t}")
        ax.set_title(
            "Per-slot marginal cost (bar color = active job)", fontsize=9, loc="left"
        )
        ax.grid(True, axis="y", linestyle=":", alpha=self.grid_alpha)

    # -----------------------------
    # Panel 3: Per-job contribution
    # -----------------------------
    def _draw_contribution_bar(
        self,
        ax,
        mdl: SingleMachinePreemptionMcf,
        blocks: list[PreemptiveBlock],
        objective: int,
        job_to_color: dict[str, tuple[float, float, float, float]],
    ) -> None:
        per_job_cost: dict[str, int] = {j: 0 for j in mdl.calJ}
        per_job_blocks: dict[str, int] = {j: 0 for j in mdl.calJ}
        for blk in blocks:
            per_job_cost[blk.job_id] += blk.cost
            per_job_blocks[blk.job_id] += 1

        items = [(j, per_job_cost[j], per_job_blocks[j]) for j in mdl.calJ if per_job_cost[j] > 0]
        items.sort(key=lambda x: x[1], reverse=True)
        if not items:
            ax.text(
                0.5,
                0.5,
                "no tardy contribution (objective = 0)",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )
            ax.set_axis_off()
            return

        # Cap to top-K and roll the tail into "others (N jobs)".
        if len(items) > self.bar_top_k:
            head = items[: self.bar_top_k]
            tail = items[self.bar_top_k :]
            tail_cost = sum(it[1] for it in tail)
            tail_blocks = sum(it[2] for it in tail)
            display_items = head + [(f"others ({len(tail)} jobs)", tail_cost, tail_blocks)]
        else:
            display_items = list(items)

        labels = [it[0] for it in display_items]
        values = [it[1] for it in display_items]
        colors = [
            job_to_color[it[0]] if it[0] in job_to_color else (0.6, 0.6, 0.6, 1.0)
            for it in display_items
        ]
        y_pos = list(range(len(display_items)))

        ax.barh(y_pos, values, color=colors, edgecolor="black", linewidth=0.4, alpha=0.9)
        denom = objective if objective > 0 else 1
        for y, (j, cost, n_blk) in zip(y_pos, display_items):
            pct = 100.0 * cost / denom
            ax.text(
                cost,
                y,
                f"  {cost} ({pct:.0f}%, {n_blk} blk)",
                va="center",
                ha="left",
                fontsize=8,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Σ c over job's blocks")
        ax.set_title(
            f"Per-job contribution (top {len(display_items)} shown; Σ all jobs = {sum(per_job_cost.values())}, objective = {objective})",
            fontsize=9,
            loc="left",
        )
        ax.grid(True, axis="x", linestyle=":", alpha=self.grid_alpha)
        # leave headroom for text annotations
        ax.set_xlim(0, max(values) * 1.25)

    # -----------------------------
    # helpers
    # -----------------------------
    def _make_color_map(
        self, job_list: list[str]
    ) -> dict[str, tuple[float, float, float, float]]:
        cmap = plt.get_cmap(self.cmap_name)
        n = max(len(job_list) - 1, 1)
        return {j: cmap(i / n) for i, j in enumerate(job_list)}
