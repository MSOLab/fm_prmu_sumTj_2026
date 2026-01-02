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
        # At this point, obj1_val are equal; define a consistent ordering on obj2_val,
        # treating a missing (None) secondary objective as greater than any concrete value.
        if self.obj2_val is None and other.obj2_val is None:
            # Equal in both objectives -> not less than
            return False
        if self.obj2_val is None:
            # self has no secondary objective, other does -> self is considered greater
            return False
        if other.obj2_val is None:
            # other has no secondary objective, self does -> self is considered less
            return True
        # Both have concrete secondary objectives -> compare them
        return self.obj2_val < other.obj2_val

    def __gt__(self, other: ObjValVector) -> bool:
        return other < self

    def __le__(self, other: ObjValVector) -> bool:
        return not self > other

    def __ge__(self, other: ObjValVector) -> bool:
        return not self < other

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ObjValVector):
            return NotImplemented
        return (
            self.obj1_val == other.obj1_val
            and self.obj2_val == other.obj2_val
        )

    def __hash__(self) -> int:
        return hash((self.obj1_val, self.obj2_val))
