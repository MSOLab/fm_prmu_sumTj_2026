import random
from typing import List, Tuple

import pytest

from flowshop_tardiness.controller.flowshop_new_acc import PermutationFlowshopEvaluator


@pytest.fixture(scope="module")
def flowshop_case() -> Tuple[List[List[int]], List[int], List[int], int]:
    """Small deterministic instance used by both tests."""
    p = [
        [3, 2, 4, 6],
        [5, 1, 3, 2],
        [4, 2, 1, 3],
    ]
    due = [11, 7, 13, 9]
    pi = [0, 2, 1]
    sigma = 3
    return p, due, pi, sigma


@pytest.fixture(scope="module")
def p(flowshop_case):
    return flowshop_case[0]


@pytest.fixture(scope="module")
def due(flowshop_case):
    return flowshop_case[1]


@pytest.fixture(scope="module")
def pi(flowshop_case):
    return flowshop_case[2]


@pytest.fixture(scope="module")
def sigma(flowshop_case):
    return flowshop_case[3]


# ---------------------------
# Helper: naive full DP for a given sequence
# ---------------------------
def naive_completion_times(p: List[List[int]], seq: List[int]) -> List[List[int]]:
    """
    Full DP table C[i][j] for permutation flowshop.
    p: processing times [m][n_jobs]
    seq: job order
    Returns: C with shape (m, len(seq)) where C[i][j] is completion time of seq[j] on machine i.
    """
    m = len(p)
    n = len(seq)
    C = [[0] * n for _ in range(m)]
    for j, job in enumerate(seq):
        for i in range(m):
            up = C[i - 1][j] if i > 0 else 0
            left = C[i][j - 1] if j > 0 else 0
            start = up if up > left else left
            C[i][j] = start + p[i][job]
    return C


def naive_sum_tardiness(p: List[List[int]], due: List[int], seq: List[int]) -> int:
    C = naive_completion_times(p, seq)
    m = len(p)
    total = 0
    for j, job in enumerate(seq):
        C_last = C[m - 1][j]
        tardy = C_last - due[job]
        if tardy > 0:
            total += tardy
    return total


def naive_sum_idle(p: List[List[int]], seq: List[int]) -> int:
    """
    Total machine idle time:
      Idle_{i,j} = S_{i,j} - C_{i,j-1}
                = max(C_{i-1,j}, C_{i,j-1}) - C_{i,j-1}
                = max(0, C_{i-1,j} - C_{i,j-1})
    """
    C = naive_completion_times(p, seq)
    m = len(p)
    n = len(seq)
    total = 0
    for j in range(n):
        for i in range(m):
            up = C[i - 1][j] if i > 0 else 0
            left = C[i][j - 1] if j > 0 else 0
            idle = up - left
            if idle > 0:
                total += idle
    return total


def insert_job(pi: List[int], sigma: int, pos: int) -> List[int]:
    """Insert sigma into pi at position pos (0..len(pi))."""
    return pi[:pos] + [sigma] + pi[pos:]


# ---------------------------
# TEST 1: Fig.10 evaluator matches naive for ALL positions
# ---------------------------
def test_fig10_matches_naive_one_instance(p, due, pi, sigma):
    """
    For a fixed instance and fixed (pi, sigma),
    compare NEW (Fig.9 + Cor.3.2 + Fig.10) against naive for every insertion position.
    """
    solver = PermutationFlowshopEvaluator(p, due)
    pre = solver.precompute(pi, sigma)

    for pos in range(len(pi) + 1):
        # i* from Cor.3.2
        i_star, _ = solver.find_i_star(pre, pos)

        # NEW evaluation (Fig.10)
        # NOTE: this assumes your calculate_OF_fig10 returns ObjValVector(total_tardy, total_idle or None)
        new_val = solver.calculate_OF_fig10(
            pi=pi,
            sigma=sigma,
            i_star=i_star,
            cp=pre.cp,
            csigma=pre.csigma,
            j1=pos,
            AOF=pre.prefix_tardy,
        )

        # NAIVE evaluation
        seq2 = insert_job(pi, sigma, pos)
        naive_tardy = naive_sum_tardiness(p, due, seq2)

        assert new_val == naive_tardy, (
            f"[Mismatch tardy] pos={pos} NEW={new_val} NAIVE={naive_tardy} seq2={seq2}"
        )


# ---------------------------
# TEST 2: best insertion from NEW equals best insertion from naive
# ---------------------------
def test_best_insertion_matches_naive_one_instance(p, due, pi, sigma):
    solver = PermutationFlowshopEvaluator(p, due)

    # NEW best
    best_pos_new, best_val_new = solver.get_best_position(pi, sigma)

    # naive best
    best_pos_naive = None
    best_val_naive = None
    for pos in range(len(pi) + 1):
        seq2 = insert_job(pi, sigma, pos)
        val = naive_sum_tardiness(p, due, seq2)
        if best_val_naive is None or val < best_val_naive:
            best_val_naive = val
            best_pos_naive = pos

    assert best_val_new == best_val_naive, (
        f"[Best value mismatch] NEW={best_val_new} NAIVE={best_val_naive}"
    )
    assert best_pos_new == best_pos_naive, (
        f"[Best pos mismatch] NEW={best_pos_new} NAIVE={best_pos_naive}"
    )


# ---------------------------
# Randomized regression runner (no pytest needed)
# ---------------------------
def run_random_tests(
    seed: int = 0,
    n_trials: int = 50,
    m_range: Tuple[int, int] = (2, 5),
    n_jobs_range: Tuple[int, int] = (4, 10),
    p_range: Tuple[int, int] = (1, 20),
):
    random.seed(seed)

    for t in range(n_trials):
        m = random.randint(*m_range)
        n_jobs = random.randint(*n_jobs_range)

        # processing times
        p = [[random.randint(*p_range) for _ in range(n_jobs)] for _ in range(m)]

        # due dates: set around typical completion scale
        # rough: sum of average processing times
        avg_p = sum(sum(row) for row in p) / (m * n_jobs)
        # make due a bit tight to generate tardiness
        due = [
            random.randint(int(avg_p * m * 0.5), int(avg_p * m * 2.0))
            for _ in range(n_jobs)
        ]

        # choose pi and sigma (sigma not in pi)
        jobs = list(range(n_jobs))
        random.shuffle(jobs)

        # pick length of pi between 1 and n_jobs-1
        L = random.randint(1, n_jobs - 1)
        pi = jobs[:L]
        sigma = jobs[L]

        # Run tests
        test_fig10_matches_naive_one_instance(p, due, pi, sigma)
        test_best_insertion_matches_naive_one_instance(p, due, pi, sigma)

        if (t + 1) % 10 == 0:
            print(f"[OK] {t + 1}/{n_trials} trials passed")

    print(f"All {n_trials} randomized trials passed ✅")


if __name__ == "__main__":
    run_random_tests(seed=0, n_trials=50)
