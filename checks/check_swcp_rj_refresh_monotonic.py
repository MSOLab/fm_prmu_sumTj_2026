import random
from itertools import permutations

from flowshop_tardiness.fm_prmu import PermutationFlowshopScheduleLite as Sched


def total_tard(stages, p, due, seq, est=None):
    s = Sched(stages, p, due)
    s.extend_jobs(seq, stage_2_est_map=est)
    return s.get_total_tardiness()


def sim_swcp(stages, p, due, pi, B, refresh):
    """refresh=False: S^R once from original incumbent (current code).
    refresh=True : S^R recomputed each step from committed+remaining (proposed fix)."""
    n = len(pi)
    sr0 = Sched(stages, p, due)
    sr0.extend_jobs(pi)
    sr0.push_back_tail_jobs_keep_tardiness(n)
    rs_static = {j: sr0.get_stage_2_start_time_map(j) for j in pi}
    committed = []
    pref = Sched(stages, p, due)
    remaining = pi[:]
    traj = []
    while remaining:
        batch = remaining[:B]
        last_in = len(remaining) <= B
        est = pref.get_stage_2_makespan_map() if committed else None
        if last_in:
            lct = None
        else:
            jnext = remaining[B]
            if refresh:
                cur = committed + remaining  # immediately-preceding schedule
                srk = Sched(stages, p, due)
                srk.extend_jobs(cur)
                srk.push_back_tail_jobs_keep_tardiness(len(cur))
                lct = srk.get_stage_2_start_time_map(jnext)
            else:
                lct = rs_static[jnext]
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
        picked = best[1] if best else list(batch)
        improved = picked != list(batch)
        step = B if improved else 1
        newly = picked[:step]
        pref.extend_jobs(newly)
        committed += newly
        cset = set(committed)
        remaining = [j for j in pi if j not in cset]
        traj.append(total_tard(stages, p, due, committed + remaining))
    return traj


random.seed(7)
trials = 8000
inc_static = inc_refresh = 0
worse_final = 0
for t in range(trials):
    n = random.randint(6, 11)
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
    ts = sim_swcp(stages, p, due, jobs, B, refresh=False)
    tr = sim_swcp(stages, p, due, jobs, B, refresh=True)
    if any(ts[i + 1] > ts[i] for i in range(len(ts) - 1)):
        inc_static += 1
    if any(tr[i + 1] > tr[i] for i in range(len(tr) - 1)):
        inc_refresh += 1
    if tr[-1] > ts[-1]:  # does refresh hurt final quality?
        worse_final += 1

print(f"trials={trials}")
print(f"  current (S^R once)   : trajectories with an increase = {inc_static}")
print(f"  proposed (refresh)   : trajectories with an increase = {inc_refresh}")
print(f"  refresh final WORSE than current final = {worse_final}")
