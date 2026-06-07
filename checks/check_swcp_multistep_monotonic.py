import random
from itertools import permutations

from flowshop_tardiness.fm_prmu import PermutationFlowshopScheduleLite as Sched

random.seed(7)


def total_tard(stages, p, due, seq, est=None):
    s = Sched(stages, p, due)
    s.extend_jobs(seq, stage_2_est_map=est)
    return s.get_total_tardiness()


def sim_swcp(stages, p, due, pi, B):
    """Faithful brute-force SW-CP. Returns list of full-obj after each commit."""
    n = len(pi)
    # S^R
    sr = Sched(stages, p, due)
    sr.extend_jobs(pi)
    sr.push_back_tail_jobs_keep_tardiness(n)
    rs = {
        j: sr.get_stage_2_start_time_map(j) for j in pi
    }  # right-justified start per job

    committed = []  # committed prefix (CP order)
    pref = Sched(stages, p, due)  # committed schedule
    remaining = pi[:]  # incumbent order
    full_traj = []
    while remaining:
        batch = remaining[:B]
        last_in = len(remaining) <= B
        est = pref.get_stage_2_makespan_map() if committed else None
        if not last_in:
            jnext = remaining[B]
            lct = rs[jnext]
        else:
            lct = None
        # 'CP': enumerate feasible batch orders, lexicographic (tardiness, sum makespan)
        best = None
        for order in permutations(batch):
            bs = Sched(stages, p, due)
            bs.extend_jobs(list(order), stage_2_est_map=est)
            msp = bs.get_stage_2_makespan_map()
            if lct is not None and any(msp[s] > lct[s] for s in stages):
                continue
            key = (bs.get_total_tardiness(), sum(msp.values()))
            if best is None or key < best[0]:
                best = (key, list(order))
        picked = best[1] if best is not None else list(batch)  # fallback incumbent
        improved = picked != list(batch)
        step = B if improved else 1
        newly = picked[:step]
        pref.extend_jobs(newly)
        committed += newly
        cset = set(committed)
        remaining = [j for j in pi if j not in cset]
        # full obj = committed + remaining(incumbent order), greedy
        full = total_tard(stages, p, due, committed + remaining)
        full_traj.append(full)
    return full_traj


incr_cases = 0
trials = 4000
for t in range(trials):
    n = random.randint(6, 10)
    c = random.randint(2, 4)
    B = random.randint(2, 3)
    stages = [f"s{i}" for i in range(c)]
    jobs = [f"j{j}" for j in range(n)]
    p = {jb: {s: random.randint(1, 9) for s in stages} for jb in jobs}
    base = Sched(stages, p, {jb: 0 for jb in jobs})
    base.extend_jobs(jobs)
    last = base.last_stage_name
    comp = {jb: base._stage_2_job_2_end_map[last][jb] for jb in jobs}
    due = {jb: comp[jb] + random.randint(-6, 10) for jb in jobs}
    traj = sim_swcp(stages, p, due, jobs, B)
    if any(traj[i + 1] > traj[i] for i in range(len(traj) - 1)):
        incr_cases += 1
        if incr_cases <= 6:
            ups = [
                (i, traj[i], traj[i + 1])
                for i in range(len(traj) - 1)
                if traj[i + 1] > traj[i]
            ]
            print(
                f"INCREASE trial={t} n={n} c={c} B={B}: {ups[:3]}  traj_head={traj[:8]}"
            )

print(f"done. trials={trials}, trajectories with an increase = {incr_cases}")
