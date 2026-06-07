import random

from flowshop_tardiness.fm_prmu import PermutationFlowshopScheduleLite as Sched

random.seed(0)


def per_job_tardiness(sched):
    last = sched.last_stage_name
    out = {}
    for j in sched._job_seq:
        c = sched._stage_2_job_2_end_map[last][j]
        d = sched._job_2_due_map[j]
        out[j] = max(0, c - d)
    return out


violations = 0
for trial in range(2000):
    n = random.randint(3, 9)
    c = random.randint(1, 4)
    stages = [f"s{i}" for i in range(c)]
    jobs = [f"j{j}" for j in range(n)]
    p = {jb: {s: random.randint(1, 9) for s in stages} for jb in jobs}
    # left-justified schedule first to get a baseline completion, then due dates
    base = Sched(stages, p, {jb: 0 for jb in jobs})
    base.extend_jobs(jobs)
    last = base.last_stage_name
    comp = {jb: base._stage_2_job_2_end_map[last][jb] for jb in jobs}
    # due dates spread around completion (some early -> slack, some tight)
    due = {jb: comp[jb] + random.randint(-5, 8) for jb in jobs}

    s = Sched(stages, p, due)
    s.extend_jobs(jobs)
    before = per_job_tardiness(s)
    total_before = sum(before.values())
    # right-justify all jobs
    s.push_back_tail_jobs_keep_tardiness(len(jobs))
    after = per_job_tardiness(s)
    total_after = sum(after.values())
    if before != after:
        violations += 1
        if violations <= 5:
            print(f"TRIAL {trial} n={n} c={c}: per-job tardiness CHANGED")
            print("  before:", before)
            print("  after :", after)
            print("  total before/after:", total_before, total_after)

print("done. trials=2000, tardiness-change violations =", violations)
