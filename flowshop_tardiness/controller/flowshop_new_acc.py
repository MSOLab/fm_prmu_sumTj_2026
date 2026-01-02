from dataclasses import dataclass
from typing import Sequence

from .obj_val_vector import ObjValVector


@dataclass
class Precomp:
    c: list[list[int]]
    """
    forward DP completion time of pi (paper's c_{ij}; size m x (k-1))
    """

    cbar: list[list[int]]
    """
    reverse DP completion time of pi (paper's \\bar{c}_{ij}; size m x (k-1))
    """

    cp: list[list[int]]
    """
    direction chosen while computing cbar (paper's cp_{ij})
    cp=0 means we chose "down" (i+1,j), cp=1 means we chose "right" (i,j+1)
    """

    csigma: list[list[int]]
    """
    completion time of σ when inserted at position pos (paper's c^{σ}_{ij})
    (pos in [0..k-1])
    """

    prefix_tardy: list[int]
    """
    AOF[pos]: sum of tardiness of prefix up to position pos (0..k-1)
    """


class PermutationFlowshopEvaluator:
    """
    NEW acceleration (Fernandez-Viagas et al., 2020) implemented in DP-table style.
    Supports evaluation of insertion moves for completion-time based objectives.
    Here we implement total tardiness sumTj.
    """

    def __init__(self, p: Sequence[Sequence[int]], due: Sequence[int]):
        """
        p: processing times [m][n_jobs]
        due: due dates [n_jobs]
        """
        self.p = p
        self.due = due
        self.m = len(p)
        self.n_jobs = len(due)

    def get_tardiness(self, job: int, completion_time: int) -> int:
        """
        Return tardiness of job given its completion time.
        """
        return completion_time - self.due[job] if completion_time > self.due[job] else 0

    def precompute(self, pi: Sequence[int], sigma: int) -> Precomp:
        """
        (As faithful as possible) Implementation of the algorithm in Fig.9.
        """
        m = self.m
        k_minus_1 = len(pi)  # |pi| = k-1
        k = k_minus_1 + 1  # insertion positions count (paper's k)

        # ============================================================
        # 1) Calculate c_ij  (forward DP on Π)  -- Fig.9 top block
        # ============================================================
        # Here we use 0-based indices:
        #   i = 0..m-1  corresponds to paper i=1..m
        #   j = 0..k-2  corresponds to paper j=1..k-1
        c = [[0] * k_minus_1 for _ in range(m)]
        for j in range(k_minus_1):  # paper j=1..k-1
            job = pi[j]
            for i in range(m):  # paper i=1..m
                up = c[i - 1][j] if i > 0 else 0  # c_{i-1,j} or c_{0,j}=0
                left = c[i][j - 1] if j > 0 else 0  # c_{i,j-1} or c_{i,0}=0
                c[i][j] = (up if up > left else left) + self.p[i][job]  # c_{ij}

        # ============================================================
        # 2) Calculate \bar{c}_{ij} and cp_{ij}  -- Fig.9 middle block
        # ============================================================
        # In paper, boundaries are:
        #   \bar{c}_{m+1,j} = 0  for all j = 1..k
        #   \bar{c}_{i,k}   = 0  for all i = 1..m
        #
        # To represent those boundaries naturally, we allocate:
        #   cbar size = (m+1) x (k)
        #
        # Indices mapping:
        #   i = 0..m-1 in our loops (paper 1..m)
        #   plus one extra row i=m  representing paper (m+1)
        #   j = 0..k-2 corresponds to paper j=1..k-1 (real jobs in Π)
        #   plus one extra col j=k-1 representing paper j=k boundary
        #
        # After computation, we will return only the "real" part:
        #   cbar_real[i][j] = cbar[i][j] for i=0..m-1, j=0..k-2
        cbar_full = [[0] * k for _ in range(m + 1)]
        cp_full = [
            [0] * k for _ in range(m + 1)
        ]  # we will return cp[i][j] for real region only

        if k_minus_1 > 0:
            # ---- base column: j = k-2  (paper j = k-1) ----
            last_j = k_minus_1 - 1
            last_job = pi[last_j]
            for i in range(m - 1, -1, -1):  # \forall i=m..1
                # (in paper) \bar{c}_{i,k-1} = \bar{c}_{i+1,k-1} + p_{i,π_{k-1}}
                cbar_full[i][last_j] = cbar_full[i + 1][last_j] + self.p[i][last_job]
                # (in paper) cp_{i,k-1} = 0
                cp_full[i][last_j] = 0

            # ---- main loop: for j = k-2 down to 1 (paper) ----
            # our j: (k-3 down to 0)
            for j in range(k_minus_1 - 2, -1, -1):
                job = pi[j]

                # (a) i = m (paper's last machine, i=m)
                i = m - 1
                # In paper: \bar{c}_{m,j} = \bar{c}_{m,j+1} + p_{m,π_j}; cp_{m,j}=1
                cbar_full[i][j] = cbar_full[i][j + 1] + self.p[i][job]
                cp_full[i][j] = 1  # because we "move right" on the bottom machine

                # (b) i = m-1 down to 1 (paper); our i=m-2..0
                for i in range(m - 2, -1, -1):
                    down = cbar_full[i + 1][j]  # \bar{c}_{i+1,j}
                    right = cbar_full[i][j + 1]  # \bar{c}_{i,j+1}
                    if down >= right:
                        cbar_full[i][j] = down + self.p[i][job]
                        cp_full[i][j] = 0
                    else:
                        cbar_full[i][j] = right + self.p[i][job]
                        cp_full[i][j] = 1

        # Extract real region (m x (k-1))
        cbar = [row[:k_minus_1] for row in cbar_full[:m]]
        cp = [row[:k_minus_1] for row in cp_full[:m]]

        # ============================================================
        # 3) Calculate c^{σ}_{ij}  (σ insertion DP for all positions)
        # ============================================================
        # In paper:
        #   c^{σ}_{0j}=0  for j=1..k
        #   c^{σ}_{ij} = max(c_{i,j-1}, c^{σ}_{i-1,j}) + p_{i,σ}
        #
        # Interpretation:
        # - We want completion time of σ on each machine i,
        #   assuming σ is inserted at each position j (1..k).
        # - "left neighbor" is the completion time of the job just before insertion point.
        #
        # In our 0-based:
        #   pos = 0..k-1   corresponds to paper j=1..k
        csigma = [[0] * k for _ in range(m)]
        for j in range(k):
            for i in range(m):
                left = c[i][j - 1] if j > 0 else 0
                up = csigma[i - 1][j] if i > 0 else 0
                # paper: max(c_{i,j-1}, c^{σ}_{i-1,j}) + p_{i,σ}
                csigma[i][j] = (left if left > up else up) + self.p[i][sigma]

        # ============================================================
        # 4) Calculate AOF_j (prefix objective)  -- Fig.9 bottom block
        # ============================================================
        # Paper shows example for total completion time:
        #   AOF_1 = c_{m,1}
        #   AOF_j = AOF_{j-1} + c_{m,j}
        #
        # In your implementation, objective1 is total tardiness:
        #   AOF_j = AOF_{j-1} + max(0, C_m(pi_j) - d_{pi_j})
        prefix_tardy = [0] * (k_minus_1 + 1)
        for t in range(1, k_minus_1 + 1):
            C_last = c[m - 1][t - 1]
            job = pi[t - 1]
            prefix_tardy[t] = prefix_tardy[t - 1] + self.get_tardiness(job, C_last)

        return Precomp(
            c=c,
            cbar=cbar,
            cp=cp,
            csigma=csigma,
            prefix_tardy=prefix_tardy,
        )

    def find_i_star(self, pre: Precomp, pos: int) -> tuple[int, int]:
        """
        Corollary 3.2 (paper):
        After inserting job σ at position pos (paper j),
        the new makespan is:

            makespan(pos) = max_{i=1..m} ( c^σ_{i,pos} + \\bar{c}_{i,pos} )

        and i* is the machine index that attains this maximum.

        Index mapping (0-based in code):
        - pos in [0..len(pi)]  (pos==len(pi) means insert at end)
        - pre.csigma[i][pos] corresponds to c^σ_{i, pos} (paper)
        - pre.cbar[i][pos] corresponds to \bar{c}_{i, pos} for the suffix starting at pi[pos]
            Only defined when pos < len(pi). If pos == len(pi), suffix is empty so \bar{c}=0.
        """
        m = self.m
        n = len(pre.c[0]) if m > 0 else 0  # n == len(pi) == k-1

        best_val = -1
        i_star = 0

        for i in range(m):
            # suffix contribution: \bar{c}_{i,pos}
            # If pos == n, there is no suffix job after σ, so suffix time is 0.
            suffix = pre.cbar[i][pos] if pos < n else 0

            # candidate critical-path length through machine i
            cand = pre.csigma[i][pos] + suffix

            # different from the paper: tie-break now prefers south (larger i)
            if cand > best_val or (cand == best_val and i > i_star):
                best_val = cand
                i_star = i

        return i_star, best_val

    def calculate_OF_fig10(
        self,
        sigma: int,
        pi: Sequence[int],
        i_star: int,
        cp: Sequence[Sequence[int]],
        csigma: Sequence[Sequence[int]],
        j1: int,
        AOF: Sequence[int],
    ) -> int:
        """(As faithful as possible) Implementation of the algorithm in Fig.10.

        Args:
            sigma (int): new job σ to insert
            pi (Sequence[int]): original sequence Π of length L (paper "Length")
            i_star (int): critical machine where the new critical path connects sigma to Π (from Cor.3.2)
            cp (Sequence[Sequence[int]]): direction table computed in Fig.9 while building cbar
            csigma (Sequence[Sequence[int]]): completion time of sigma if inserted at each position j (Fig.9)
            j1 (int): insertion position (0-based). sigma is placed before pi[j1] (if j1 < L), else at end.
            AOF (Sequence[int]): prefix objective for Π (Fig.9)

        Returns:
            int: objective value for Π' = insert(sigma at j1) computed using Fig.10 logic.
        """

        m = self.m
        L = len(pi)  # paper: Length := Length of Π

        # -----------------------------
        # Allocate arrays consistent with pseudocode
        # -----------------------------
        # Load[i] corresponds to Load_i in paper (i=1..m)
        Load = [0] * m

        # C[i][t] corresponds to C_{i,t} in paper (i=1..m, t is position in Π')
        # Π' length is L+1 after insertion.
        # We'll store only what Fig.10 writes:
        #   - C_{i, j1} initial column for sigma
        #   - then C_{i, j1+1}, C_{i, j1+2}, ... as needed
        C = [[0] * (L + 1) for _ in range(m)]

        # -----------------------------
        # 1) Initialize Load and C for sigma at position j1
        # -----------------------------
        # Paper:
        #   Load_i = c^σ_{ij1}  for all i=1..m
        #   C_{ij1} = Load_i    for all i=1..m
        #
        # In our 0-based:
        #   position j1 in Π' is sigma's position.
        for i in range(m):
            Load[i] = csigma[i][j1]  # completion time of sigma on machine i
            C[i][j1] = Load[i]  # store it as C_{i, sigma_pos}

        # -----------------------------
        # 2) Special handling if sigma is NOT inserted at the end
        # -----------------------------
        # Paper:
        #   if j1 <= Length then
        #       Load_{i*} = Load_{i*} + p_{i*, π_{j1}}
        #       C_{i*, j1+1} = Load_{i*}
        #   end
        #
        # Interpretation:
        # If there is a job immediately after sigma (i.e., pi[j1] exists),
        # then on the critical machine i*, we can immediately "start" building
        # the next job's completion time on that boundary.
        #
        if j1 < L:
            next_job = pi[j1]
            Load[i_star] = Load[i_star] + self.p[i_star][next_job]
            C[i_star][j1 + 1] = Load[i_star]

        # -----------------------------
        # 3) Main loop over suffix jobs: for j = j1 to Length do
        # -----------------------------
        # Paper's j runs over original Π indices that are shifted right by insertion.
        #
        # In our 0-based:
        #   j refers to index in original pi, where pi[j] is the job after sigma when j=j1.
        #
        # Important: Fig.10 updates:
        #   - if cp_{i*,j} == 1: treat as "horizontal" case (boundary stays)
        #   - else: "vertical" case (boundary i* moves down until cp becomes 1)
        #
        for j in range(j1, L):
            job = pi[j]

            # CASE 1: cp_{i*,j} == 1
            if cp[i_star][j] == 1:
                # Paper:
                # for i = i*+1 to m do
                #     Load_i = max(Load_{i-1}, Load_i) + p_{i, π_j}
                #     C_{i, j+1} = Load_i
                # end
                #
                # Here, i* itself already updated earlier for this job (at "if j+1 <= Length" part)
                # so we start from i = i_star + 1.
                for i in range(i_star + 1, m):
                    # Load[i] currently plays role of "left" (same machine, previous job completion)
                    # Load[i-1] is "up" (previous machine, same job completion)
                    if Load[i - 1] > Load[i]:
                        Load[i] = Load[i - 1]
                    Load[i] += self.p[i][job]
                    C[i][j + 1] = Load[i]

                # Paper:
                # if j+1 <= Length then
                #     Load_{i*} = Load_{i*} + p_{i*, π_{j+1}}
                #     C_{i*, j+2} = Load_{i*}
                # end
                #
                if (j + 1) < L:
                    next_job = pi[j + 1]
                    Load[i_star] = Load[i_star] + self.p[i_star][next_job]
                    C[i_star][j + 2] = Load[i_star]

            # CASE 2: cp_{i*,j} == 0
            else:
                # Paper:
                # Load_{i*+1} = Load_{i*} + p_{i*+1, π_j}
                # C_{i*+1, j+1} = Load_{i*+1}
                # i*++
                #
                # This means the boundary moves DOWN by 1 machine for current job,
                # because reverse DP says critical suffix prefers "down" here.
                if i_star + 1 < m:
                    Load[i_star + 1] = Load[i_star] + self.p[i_star + 1][job]
                    C[i_star + 1][j + 1] = Load[i_star + 1]
                    i_star += 1

                # Paper:
                # while cp_{i*,j} = 0 do
                #     Load_{i*+1} = Load_{i*} + p_{i*+1, π_j}
                #     C_{i*+1, j+1} = Load_{i*+1}
                #     i*++
                # end
                #
                # Keep moving boundary down while cp indicates vertical continuation.
                while i_star < m - 1 and cp[i_star][j] == 0:
                    Load[i_star + 1] = Load[i_star] + self.p[i_star + 1][job]
                    C[i_star + 1][j + 1] = Load[i_star + 1]
                    i_star += 1

                # Paper:
                # for i = i*+1 to m do
                #     Load_i = max(Load_{i-1}, Load_i) + p_{i, π_j}
                # end
                #
                for i in range(i_star + 1, m):
                    if Load[i - 1] > Load[i]:
                        Load[i] = Load[i - 1]
                    Load[i] += self.p[i][job]
                    C[i][j + 1] = Load[i]

                # Paper:
                # if j+1 <= Length then
                #     Load_{i*} = Load_{i*} + p_{i*, π_{j+1}}
                #     C_{i*, j+2} = Load_{i*}
                # end
                #
                if (j + 1) < L:
                    next_job = pi[j + 1]
                    Load[i_star] = Load[i_star] + self.p[i_star][next_job]
                    C[i_star][j + 2] = Load[i_star]

        # -----------------------------
        # 4) Calculate objective function
        # -----------------------------
        # σ tardiness (σ is at position j1 in Π')
        total_tardiness = self.get_tardiness(sigma, C[m - 1][j1])

        # suffix jobs tardiness (these are original pi[j1..L-1], shifted by +1 position)
        for j in range(j1, L):
            # pi[j] now sits at position (j+1) in Π'
            job = pi[j]
            total_tardiness += self.get_tardiness(job, C[m - 1][j + 1])

        # add unchanged prefix objective before insertion point
        total_tardiness += AOF[j1]

        return total_tardiness

    def get_best_position(
        self, pi: Sequence[int], sigma: int, tie_breaker: str = "default"
    ) -> tuple[int, int]:
        """Return (best_pos, best_OF) for inserting sigma into pi.

        This is the top-level driver:
          - Fig.9 precompute
          - Cor.3.2 to find i*
          - Fig.10 to compute objective (sumTj)

        Args:
            pi (Sequence[int]): original job sequence
            sigma (int): new job to insert
            tie_breaker (str): tie breaking strategy.
                If "makespan", store makespan as secondary objective.

        Returns:
            tuple[int, int]: (best_pos, best_total_tardiness)
        """
        pre = self.precompute(pi, sigma)

        best_pos = 0
        best_obj_vals: ObjValVector | None = None

        # Try all insertion positions pos in [0..len(pi)]
        for pos in range(len(pi) + 1):
            i_star, makespan = self.find_i_star(pre, pos)

            sum_Tj = self.calculate_OF_fig10(
                pi=pi,
                sigma=sigma,
                i_star=i_star,
                cp=pre.cp,
                csigma=pre.csigma,
                j1=pos,
                AOF=pre.prefix_tardy,
            )
            if tie_breaker == "makespan":
                obj_vals = ObjValVector(sum_Tj, makespan)
            else:
                obj_vals = ObjValVector(sum_Tj)

            if best_obj_vals is None or obj_vals < best_obj_vals:
                best_pos = pos
                best_obj_vals = obj_vals

        if best_obj_vals is None:
            raise ValueError("No insertion positions evaluated.")
        return best_pos, best_obj_vals.obj1_val
