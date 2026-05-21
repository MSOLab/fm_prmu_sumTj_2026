from __future__ import annotations

from dataclasses import dataclass

from ortools.graph.python.min_cost_flow import SimpleMinCostFlow
from schore.parameters_examples.shop.flow import FlowshopDuedateParameters


@dataclass
class McfSolution:
    is_optimal: bool
    objective_value: int
    x: dict[str, dict[int, int]]  # flow on (j,t): 0/1 (정수)
    job_2_completion_time: dict[str, int]
    job_completion_sequence: list[str]


@dataclass
class PreemptiveBlock:
    job_id: str
    start_t: int  # first unit slot (inclusive)
    end_t: int  # last unit slot (inclusive)
    cost: int  # Σ c_{j,t} over the block
    tardy_cost: int  # Σ c_{j,t} restricted to t > d_j (== cost in current formulation)


class SingleMachinePreemptionMcf:
    """
    원래의 시간지수 모델을 그대로 네트워크(최소비용흐름)로 변환한 구현.
    - 노드:
        source -> job(j) -> time(t) -> sink
    - 용량:
        source->job : p_j
        job->time  : 1 (t > r_j 인 시간만)
        time->sink : 1
    - 비용:
        c_{j,t} = ceil( max(0, (t - d_j) / p_j) )
                 = 0 if t <= d_j else ceil((t - d_j)/p_j)
      (정수, 단위흐름 비용)
    이 네트워크는 원래 LP와 동등하며, LP의 최적값 == MCF의 최적값(=LB) 입니다.
    """

    name: str
    calJ: list[str]
    p: dict[str, int]
    r: dict[str, int]
    d: dict[str, int]
    t_max: int
    calT: list[int]
    c: dict[str, dict[int, int]]

    mcf: SimpleMinCostFlow | None
    source_id: int
    sink_id: int
    job_node_id: dict[str, int]
    time_node_id: dict[int, int]
    arc_index_job_time: dict[tuple[str, int], int]

    # Results
    status_optimal: bool
    opt_cost: int

    def __init__(self):
        self.name = "SingleMachinePreemptionMcf"
        self.calJ = []
        self.p = {}
        self.r = {}
        self.d = {}
        self.t_max = 0
        self.calT = []
        self.c = {}

        # OR-Tools 객체
        self.mcf = None

        # 내부 매핑
        self.source_id = 0
        self.sink_id = 1
        self.job_node_id = {}
        self.time_node_id = {}
        self.arc_index_job_time = {}

        # 결과
        self.status_optimal = False
        self.opt_cost = 0

    @classmethod
    def from_instance(
        cls, instance: FlowshopDuedateParameters
    ) -> SingleMachinePreemptionMcf:
        obj = cls()
        obj.name = f"{cls.__name__}_{instance.name}"
        obj._define_parameters(instance)
        obj._build_mcf()
        return obj

    # -----------------------------
    # Model construction
    # -----------------------------
    def _define_parameters(self, instance: FlowshopDuedateParameters) -> None:
        self.calJ = instance.job_id_list
        self.p = instance.get_job_2_p_map(instance.stage_id_list[-1])
        self.r = instance.get_job_2_p_sum_except_last_stage()
        self.d = instance.job_2_duedate_map

        # T = max r_j + sum p_j
        self.t_max = max(self.r.values()) + sum(self.p.values())
        self.calT = list(range(1, self.t_max + 1))

        # 비용 계수(정수): c_{j,t} = 0 if t <= d_j else ceil((t - d_j)/p_j)
        self.c = {
            j: {
                t: 0
                if t <= self.d[j]
                else ((t - self.d[j] + self.p[j] - 1) // self.p[j])
                for t in self.calT
            }
            for j in self.calJ
        }

    def _build_mcf(self) -> None:
        mcf = SimpleMinCostFlow()

        # 노드 번호 배정
        self.source_id = 0
        self.sink_id = 1

        # job 노드: 2 .. 2+|J|-1
        self.job_node_id = {j: 2 + i for i, j in enumerate(self.calJ)}
        # time 노드: 연속으로 이어서
        self.time_node_id = {t: 2 + len(self.calJ) + i for i, t in enumerate(self.calT)}

        total_supply = sum(self.p.values())

        # source -> job (용량 p_j, 비용 0)
        for j in self.calJ:
            mcf.add_arc_with_capacity_and_unit_cost(
                self.source_id, self.job_node_id[j], int(self.p[j]), 0
            )

        # job -> time (용량 1, 비용 c_{j,t}), 단 t > r_j 인 시간만 허용
        # (릴리즈 이전은 간선 생성하지 않음)
        self.arc_index_job_time = {}
        for j in self.calJ:
            rj = self.r[j]
            for t in self.calT:
                if t <= rj:
                    continue
                arc_idx = mcf.add_arc_with_capacity_and_unit_cost(
                    self.job_node_id[j], self.time_node_id[t], 1, self.c[j][t]
                )
                # (j,t) -> arc index 매핑 저장 (후에 해 추출에 사용)
                self.arc_index_job_time[(j, t)] = arc_idx

        # time -> sink (용량 1, 비용 0)
        for t in self.calT:
            mcf.add_arc_with_capacity_and_unit_cost(
                self.time_node_id[t], self.sink_id, 1, 0
            )

        # 공급/수요 설정
        mcf.set_node_supply(self.source_id, int(total_supply))
        mcf.set_node_supply(self.sink_id, -int(total_supply))
        # 나머지 노드(작업/시간)는 0 (기본값) — transit node

        self.mcf = mcf

    # -----------------------------
    # Solve & extract
    # -----------------------------
    def solve(self) -> None:
        assert self.mcf is not None
        status = self.mcf.solve()
        self.status_optimal = status == SimpleMinCostFlow.Status.OPTIMAL
        if self.status_optimal:
            self.opt_cost = self.mcf.optimal_cost()
        else:
            self.opt_cost = 0

    def is_optimal(self) -> bool:
        return self.status_optimal

    def get_obj_value(self) -> int:
        return self.opt_cost

    def get_variable_value_dict(self) -> dict[str, dict[int, int]]:
        """
        x_{j,t} = job->time 간선의 흐름 값(0/1).
        """
        assert self.mcf is not None
        value_dict: dict[str, dict[int, int]] = {j: {} for j in self.calJ}
        for (j, t), arc_idx in self.arc_index_job_time.items():
            flow = self.mcf.flow(arc_idx)
            if flow:  # 0이면 생략해도 됨
                value_dict[j][t] = int(flow)
            else:
                value_dict[j][t] = 0
        return value_dict

    def get_job_2_start_time_map(self) -> dict[str, int]:
        """
        각 작업의 시작시점: 흐름이 1인 가장 작은 t 를 start로 채택.
        (단위시간 슬롯 모델이므로 자연스러운 정의)
        """
        x = self.get_variable_value_dict()
        start: dict[str, int] = {}
        for j in self.calJ:
            # flow가 있는 최소 t
            ts = [t for t, val in x[j].items() if val > 0.5]
            start[j] = min(ts) if ts else 0
        return start

    def get_job_start_sequence(self) -> list[str]:
        start = self.get_job_2_start_time_map()
        return sorted(self.calJ, key=lambda j: start[j])

    def get_job_2_completion_time_map(self) -> dict[str, int]:
        """
        각 작업의 완료시점: 흐름이 1인 가장 큰 t 를 completion으로 채택.
        (단위시간 슬롯 모델이므로 자연스러운 정의)
        """
        x = self.get_variable_value_dict()
        comp: dict[str, int] = {}
        for j in self.calJ:
            # flow가 있는 최대 t
            ts = [t for t, val in x[j].items() if val > 0.5]
            comp[j] = max(ts) if ts else 0
        return comp

    def get_job_completion_sequence(self) -> list[str]:
        comp = self.get_job_2_completion_time_map()
        return sorted(self.calJ, key=lambda j: comp[j])

    def get_job_2_average_time_map(self) -> dict[str, float]:
        """
        각 작업의 평균 처리시점: 흐름이 1인 t 값들의 평균.
        """
        x = self.get_variable_value_dict()
        avg: dict[str, float] = {}
        for j in self.calJ:
            ts = [t for t, val in x[j].items() if val > 0.5]
            avg[j] = sum(ts) / len(ts) if ts else 0.0
        return avg

    def get_job_average_sequence(self) -> list[str]:
        avg = self.get_job_2_average_time_map()
        return sorted(self.calJ, key=lambda j: avg[j])

    def get_blocks(self) -> list[PreemptiveBlock]:
        """단위 슬롯 흐름을 같은 job의 연속 구간(block) 단위로 묶는다.

        c_{j,t} 가 t <= d_j 에서 0이므로 cost 와 tardy_cost 는 동일하지만,
        시각화에서 tardy 영역을 구분 표시하기 위해 별도 계산해 둔다.
        """
        x = self.get_variable_value_dict()
        # 시간 t -> 처리 중인 job (없으면 None)
        slot_job: dict[int, str | None] = {t: None for t in self.calT}
        for j in self.calJ:
            for t, val in x[j].items():
                if val > 0.5:
                    slot_job[t] = j

        blocks: list[PreemptiveBlock] = []
        cur_j: str | None = None
        cur_start: int = 0
        cur_cost: int = 0
        cur_tardy: int = 0

        def flush(end_t: int) -> None:
            if cur_j is not None:
                blocks.append(
                    PreemptiveBlock(
                        job_id=cur_j,
                        start_t=cur_start,
                        end_t=end_t,
                        cost=cur_cost,
                        tardy_cost=cur_tardy,
                    )
                )

        for t in self.calT:
            j = slot_job[t]
            if j is None:
                flush(t - 1)
                cur_j = None
                cur_cost = 0
                cur_tardy = 0
                continue
            if j != cur_j:
                flush(t - 1)
                cur_j = j
                cur_start = t
                cur_cost = 0
                cur_tardy = 0
            c_jt = self.c[j][t]
            cur_cost += c_jt
            if t > self.d[j]:
                cur_tardy += c_jt
        flush(self.calT[-1] if self.calT else 0)
        return blocks

    def extract_solution(self) -> McfSolution:
        return McfSolution(
            is_optimal=self.is_optimal(),
            objective_value=self.get_obj_value(),
            x=self.get_variable_value_dict(),
            job_2_completion_time=self.get_job_2_completion_time_map(),
            job_completion_sequence=self.get_job_completion_sequence(),
        )
