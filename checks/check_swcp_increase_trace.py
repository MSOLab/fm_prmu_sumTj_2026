import random
from itertools import permutations

from flowshop_tardiness.fm_prmu import PermutationFlowshopScheduleLite as Sched

random.seed(7)
# advance RNG to regenerate trial 764's instance exactly
target = 764
for t in range(target + 1):
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
    if t != target:
        continue
    print(f"TRIAL {t}: n={n} c={c} B={B}")
    print("jobs(incumbent):", jobs)
    print("p:", {j: p[j] for j in jobs})
    print("due:", due)
    print("incumbent per-job completion(last):", comp)
    pi = jobs
    sr = Sched(stages, p, due)
    sr.extend_jobs(pi)
    sr.push_back_tail_jobs_keep_tardiness(n)
    rs = {j: sr.get_stage_2_start_time_map(j) for j in pi}
    print("S^R rs (right-justified start) per job:")
    for j in pi:
        print("   ", j, rs[j])
    print(
        "incumbent full tardiness =",
        Sched(stages, p, due).__class__
        and (lambda s: (s.extend_jobs(pi), s.get_total_tardiness())[1])(
            Sched(stages, p, due)
        ),
    )

    committed = []
    pref = Sched(stages, p, due)
    remaining = pi[:]
    it = 0
    while remaining:
        it += 1
        batch = remaining[:B]
        last_in = len(remaining) <= B
        est = pref.get_stage_2_makespan_map() if committed else None
        lct = None if last_in else rs[remaining[B]]
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
        full_sched = Sched(stages, p, due)
        full_sched.extend_jobs(committed + remaining)
        full = full_sched.get_total_tardiness()
        msp_after = pref.get_stage_2_makespan_map()
        viol = (
            []
            if lct is None
            else [(s, msp_after[s], lct[s]) for s in stages if msp_after[s] > lct[s]]
        )
        print(
            f"it{it}: batch={batch} picked={picked} improved={improved} commit={newly}"
        )
        print(f"      est={est} lct={lct}")
        print(f"      committed_makespan_after={msp_after}  window_violations={viol}")
        print(f"      committed={committed}")
        print(f"      remaining={remaining}  FULL={full}")
