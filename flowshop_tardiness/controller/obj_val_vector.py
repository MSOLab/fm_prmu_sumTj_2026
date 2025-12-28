from __future__ import annotations


class ObjValVector:
    obj1_val: int
    obj2_val: int | None = None

    def __init__(self, obj1_val: int, obj2_val: int | None = None):
        self.obj1_val = obj1_val
        self.obj2_val = obj2_val

    def __lt__(self, other: ObjValVector) -> bool:
        if self.obj1_val != other.obj1_val:
            return self.obj1_val < other.obj1_val
        if self.obj2_val is not None and other.obj2_val is not None:
            return self.obj2_val < other.obj2_val
        return False
