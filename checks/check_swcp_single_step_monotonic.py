import random
from itertools import permutations

from flowshop_tardiness.fm_prmu import PermutationFlowshopScheduleLite as Sched

random.seed(1)


def build(stages, p, due, seq, est=None):
    s = Sched(stages, p, due)
    s.extend_jobs(seq, stage_2_est_map=est)
    return s


def total_tard(stages, p, due, seq):
    return build(stages, p, due, seq).get_total_tardiness()


breaks = 0
checked = 0
for trial in range(20000):
    n = random.randint(5, 9)
    c = random.randint(1, 4)
    B = random.randint(2, min(4, n - 2))  # batch size, leave >=1 tail job
    pfx = random.randint(0, n - B - 1)  # prefix length, ensure >=1 job after batch
    stages = [f"s{i}" for i in range(c)]
    jobs = [f"j{j}" for j in range(n)]
    p = {jb: {s: random.randint(1, 9) for s in stages} for jb in jobs}
    base = build(stages, p, {jb: 0 for jb in jobs}, jobs)
    last = base.last_stage_name
    comp = {jb: base._stage_2_job_2_end_map[last][jb] for jb in jobs}
    due = {jb: comp[jb] + random.randint(-6, 10) for jb in jobs}

    pi = jobs[:]  # incumbent permutation (identity)
    orig_full = total_tard(stages, p, due, pi)

    # S^R via push_back on full sequence
    sr = build(stages, p, due, pi)
    sr.push_back_tail_jobs_keep_tardiness(n)
    jnext = pi[pfx + B]  # job right after the batch
    lct = sr.get_stage_2_start_time_map(jnext)  # rs per stage

    prefix_seq = pi[:pfx]
    batch = pi[pfx : pfx + B]
    tail = pi[pfx + B :]
    # EST = prefix makespan
    pref_sched = build(stages, p, due, prefix_seq) if prefix_seq else None
    est = pref_sched.get_stage_2_makespan_map() if pref_sched else None

    # enumerate window-feasible batch orders (greedy batch makespan <= lct per stage)
    best = None  # (batch_tard, sum_makespan, order, full)
    for order in permutations(batch):
        bs = build(stages, p, due, list(order), est=est)
        msp = bs.get_stage_2_makespan_map()
        if any(msp[s] > lct[s] for s in stages):
            continue
        b_tard = bs.get_total_tardiness()
        sum_msp = sum(msp.values())
        full_seq = prefix_seq + list(order) + tail
        full = total_tard(stages, p, due, full_seq)
        key = (b_tard, sum_msp)
        if best is None or key < best[0]:
            best = (key, order, full)
    checked += 1
    if best is None:
        continue  # incumbent order should always be feasible; skip if not
    _, order, picked_full = best
    if picked_full > orig_full:
        breaks += 1
        if breaks <= 8:
            print(
                f"BREAK trial={trial} n={n} c={c} B={B} pfx={pfx}: orig_full={orig_full} picked_full={picked_full} (+{picked_full - orig_full})"
            )
            print(
                f"   batch incumbent={batch} picked={list(order)} jnext={jnext} lct={lct}"
            )

print(f"done. checked={checked}, guarantee BREAKS={breaks}")
