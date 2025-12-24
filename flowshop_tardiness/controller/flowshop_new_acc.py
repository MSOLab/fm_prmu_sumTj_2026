from dataclasses import dataclass


@dataclass
class Precomp:
    c: list[list[int]]
    """
    forward DP for pi (size m x (k-1))
    """

    cbar: list[list[int]]
    """
    reverse DP for pi (size m x (k-1))
    """

    cp: list[list[int]]
    """
    critical path direction indicator for pi
    cp[i][j] = 1 if CP goes horizontally at (i,j), 0 if it goes vertically
    """

    csigma: list[list[int]]
    """
    insertion DP for sigma: csigma[i][jpos] where jpos in [0..k-1] (k positions)
    """

    prefix_obj1: list[int]
    """
    prefix objective 1: AOF[jpos] = sum of tardiness of jobs up to position jpos-1 after insertion point handling
    We'll store prefix tardiness of original pi positions for convenience
    """

    prefix_obj2: list[int] | None = None
    """
    prefix objective 2: placeholder for potential future use
    """


class FlowshopNewAcceleration:
    """
    NEW acceleration (Fernandez-Viagas et al., 2020) implemented in DP-table style.
    Supports evaluation of insertion moves for completion-time based objectives.
    Here we implement total tardiness sumTj.
    """

    def __init__(self, p: list[list[int]], due: list[int]):
        """
        p: processing times [m][n_jobs]
        due: due dates [n_jobs]
        """
        self.p = p
        self.due = due
        self.m = len(p)
        self.n_jobs = len(due)

    # --------------------------
    # Fig.9 equivalent
    # --------------------------
    def precompute(self, pi: list[int], sigma: int) -> Precomp:
        """
        Compute:
        - c forward DP for pi
        - cbar reverse DP for pi
        - cp direction flags for pi's critical path
        - csigma insertion DP for sigma into every position j
        - prefix tardy for pi (to reuse AOF-like)
        """
        m = self.m
        k_minus_1 = len(pi)  # |pi| = k-1
        k = k_minus_1 + 1  # insertion positions count

        # ---- forward DP c[i][j] for pi ----
        # c dimension: m x k_minus_1
        c = [[0] * k_minus_1 for _ in range(m)]
        for j in range(k_minus_1):
            job = pi[j]
            for i in range(m):
                up = c[i - 1][j] if i > 0 else 0
                left = c[i][j - 1] if j > 0 else 0
                c[i][j] = max(up, left) + self.p[i][job]

        # ---- reverse DP cbar[i][j] for pi ----
        # cbar[i][j] = longest path from operation (i,j) to sink (m-1, k_minus_1-1) in reverse graph
        cbar = [[0] * k_minus_1 for _ in range(m)]
        for j in range(k_minus_1 - 1, -1, -1):
            job = pi[j]
            for i in range(m - 1, -1, -1):
                down = cbar[i + 1][j] if i < m - 1 else 0
                right = cbar[i][j + 1] if j < k_minus_1 - 1 else 0
                cbar[i][j] = max(down, right) + self.p[i][job]

        # ---- compute cp direction flags for pi critical path ----
        # We reconstruct one critical path of pi by walking from (m-1,k-1) backwards:
        # If c[i][j] == c[i][j-1] + p => came from left (horizontal edge)
        # else came from up (vertical edge)
        # cp[i][j] is defined in paper as 1 when no slack between (i,j) and (i,j+1) => horizontal continuation
        # We'll mark cp at each node as 1 if forward DP decision prefers left (horizontal), else 0 (vertical)
        cp = [[0] * k_minus_1 for _ in range(m)]
        if k_minus_1 > 0:
            i = m - 1
            j = k_minus_1 - 1
            while not (i == 0 and j == 0):
                job = pi[j]
                from_left = (
                    j > 0 and c[i][j] == (c[i][j - 1] if j > 0 else 0) + self.p[i][job]
                )
                from_up = (
                    i > 0 and c[i][j] == (c[i - 1][j] if i > 0 else 0) + self.p[i][job]
                )

                # If tie, choose horizontal to get "lower" path effect (often beneficial)
                if from_left and from_up:
                    from_left = True

                if from_left:
                    cp[i][j - 1] = (
                        1 if j - 1 >= 0 else 0
                    )  # cp at previous node indicates horizontal move
                    j -= 1
                else:
                    # vertical move
                    if i - 1 >= 0:
                        cp[i - 1][j] = 0
                    i -= 1

        # ---- compute csigma[i][pos] for sigma inserted at each position pos in [0..k-1] ----
        # csigma dimension m x k
        # This is essentially forward DP for sigma as if sigma is the pos-th job,
        # using c as "left context" (prefix completion times) and csigma as "sigma row dependency"
        csigma = [[0] * k for _ in range(m)]
        for pos in range(k):
            for i in range(m):
                # left completion time: completion at machine i of job before insertion point
                left = c[i][pos - 1] if (pos - 1) >= 0 and (pos - 1) < k_minus_1 else 0
                up = csigma[i - 1][pos] if i > 0 else 0
                csigma[i][pos] = max(left, up) + self.p[i][sigma]

        # ---- prefix tardiness for original pi (AOF-like) ----
        prefix_tardy = [0] * (
            k_minus_1 + 1
        )  # prefix_tardy[t] = tardiness sum for first t jobs of pi
        for t in range(1, k_minus_1 + 1):
            C_last = c[m - 1][t - 1]
            job = pi[t - 1]
            prefix_tardy[t] = prefix_tardy[t - 1] + max(0, C_last - self.due[job])

        # ---- prefix machine idle time for original pi (prefix_obj2) ----
        # prefix_idle[t] = sum of machine idle times considering first t jobs of pi
        prefix_idle = [0] * (k_minus_1 + 1)

        # helper to get completion time C[i][j] for pi with safe boundaries
        # c[i][j] exists for i=0..m-1, j=0..k_minus_1-1
        # define:
        #   C_{i-1,j} is 0 when i==0
        #   C_{i,j-1} is 0 when j==0
        for t in range(1, k_minus_1 + 1):
            j = t - 1  # column index in c for the t-th job
            idle_sum_this_job = 0
            for i in range(m):
                up = c[i - 1][j] if i > 0 else 0  # C_{i-1,j}
                left = c[i][j - 1] if j > 0 else 0  # C_{i,j-1}
                idle_sum_this_job += max(0, up - left)  # Idle_{i,j}
            prefix_idle[t] = prefix_idle[t - 1] + idle_sum_this_job

        return Precomp(
            c=c,
            cbar=cbar,
            cp=cp,
            csigma=csigma,
            prefix_obj1=prefix_tardy,
            prefix_obj2=prefix_idle,
        )

    # --------------------------
    # Corollary 3.2: find i*
    # --------------------------
    def find_i_star(self, pre: Precomp, pos: int) -> tuple[int, int]:
        """
        Return (i_star, makespan_after_insertion).
        For insertion at position pos:
            makespan = max_i ( csigma[i][pos] + cbar[i][pos] )
        Careful with cbar indexing when pos==k-1 (insert at end): then suffix is empty => cbar = 0.
        """
        m = self.m
        k_minus_1 = len(pre.c[0]) if self.m > 0 else 0  # size of pi
        # define suffix cbar value at insertion boundary:
        # when inserting at pos, the job from original pi that comes at that position is pi[pos]
        # its operation corresponds to column pos in original pi.
        # if pos == k_minus_1, suffix is empty -> treat cbar = 0
        best_val = -1
        i_star = 0
        for i in range(m):
            suffix = pre.cbar[i][pos] if pos < k_minus_1 else 0
            val = pre.csigma[i][pos] + suffix
            if val > best_val:
                best_val = val
                i_star = i
        return i_star, best_val

    # --------------------------
    # Fig.10 equivalent (sumTj)
    # --------------------------
    def eval_insertion_sumTj(
        self, pi: list[int], sigma: int, pos: int, pre: Precomp | None = None
    ) -> int:
        """
        Evaluate sumTj after inserting sigma into pi at position pos (0-based),
        using NEW acceleration logic (critical path guided SW computations).
        """
        if pre is None:
            pre = self.precompute(pi, sigma)

        m = self.m
        k_minus_1 = len(pi)
        k = k_minus_1 + 1  # new length after insertion
        assert 0 <= pos <= k_minus_1

        # Find i* (critical machine connecting sigma to suffix)
        i_star, _ = self.find_i_star(pre, pos)

        # ---- objective starts with prefix part of original pi before insertion ----
        # prefix before insertion includes first 'pos' jobs from pi (positions 0..pos-1)
        OF = pre.prefix_obj1[pos]

        # ---- add tardiness contribution of sigma itself ----
        C_sigma_last = pre.csigma[m - 1][pos]
        OF += max(0, C_sigma_last - self.due[sigma])

        # If insertion at end, nothing more to update
        if pos == k_minus_1:
            return OF

        # ---- Now handle suffix jobs (original pi[pos .. end]) ----
        # We'll simulate completion times for suffix jobs but guided by cp and i_star,
        # intending to skip NE computations. The paper's Fig.10 does this via Load_i
        # and advancing i_star when cp indicates vertical moves.
        #
        # We will keep an array Load[i] representing current completion time at machine i
        # for the "current job in suffix" as we process them.
        #
        # Initialize Load as completion times of sigma at each machine i.
        Load = [pre.csigma[i][pos] for i in range(m)]

        # We process each job in suffix in order (these become positions pos+1 .. k-1)
        # j_pi is the index in original pi
        j_pi = pos

        # current machine pointer along the critical path boundary
        i_cp = i_star

        while j_pi < k_minus_1:
            job = pi[j_pi]

            # Update completion times for this suffix job starting from machine i_cp
            # For machines < i_cp, the completion time is unaffected by SW rule;
            # For machines >= i_cp we update with standard DP using current Load.
            # This matches the paper's idea: only compute SW of critical path.
            for i in range(i_cp, m):
                up = Load[i - 1] if i > 0 else 0
                left = Load[i]
                Load[i] = max(up, left) + self.p[i][job]

            C_last = Load[m - 1]
            OF += max(0, C_last - self.due[job])

            # Move to next suffix job
            j_pi += 1
            if j_pi >= k_minus_1:
                break

            # Decide how the critical path boundary moves using cp of original pi.
            # cp is defined on original columns; at column j_pi-1 we inserted before that column.
            # We use cp[i_cp][j_pi-1] to decide:
            #   cp==1 -> horizontal (stay on same machine boundary)
            #   cp==0 -> vertical (move boundary down: i_cp += 1 potentially multiple times)
            col = j_pi - 1
            if col < 0:
                col = 0
            if col >= k_minus_1:
                col = k_minus_1 - 1

            if pre.cp[i_cp][col] == 1:
                # horizontal: boundary stays, but next job has to "enter" with current Load
                # nothing special needed; continue
                pass
            else:
                # vertical: boundary moves to lower machines until cp==1 or until last machine
                while i_cp < m - 1 and pre.cp[i_cp][col] == 0:
                    i_cp += 1

        return OF

    # --------------------------
    # Convenience: evaluate all positions, return best
    # --------------------------
    def best_insertion_sumTj(self, pi: list[int], sigma: int) -> tuple[int, int]:
        """
        Return (best_pos, best_OF) for inserting sigma into pi.
        """
        pre = self.precompute(pi, sigma)
        best_OF = None
        best_pos = 0
        for pos in range(len(pi) + 1):
            OF = self.eval_insertion_sumTj(pi, sigma, pos, pre=pre)
            if best_OF is None or OF < best_OF:
                best_OF = OF
                best_pos = pos
        return best_pos, best_OF
