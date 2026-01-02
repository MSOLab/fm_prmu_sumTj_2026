from collections import deque
from itertools import islice
from typing import Any, Iterator, Iterable


def window_slide_over_list(
    iterable: Iterable[Any], n: int
) -> Iterator[tuple[Any, ...]]:
    """Generate a sliding window of width n over data from the iterable.

    Args:
        iterable (Iterable[Any]): input iterable
        n (int): window size

    Raises:
        ValueError: If the window size n is not positive.
        ValueError: If the window size n is greater than the length of the iterable.

    Yields:
        tuple[Any, ...]: tuples of length n, each representing the current window
    """
    if n <= 0:
        raise ValueError("Window size n must be positive.")

    it = iter(iterable)
    window = deque(islice(it, n), maxlen=n)
    if len(window) < n:
        raise ValueError(
            "Window size n must not be greater than the length of the iterable."
        )

    while True:
        yield tuple(window)
        try:
            window.append(next(it))
        except StopIteration:
            break


if __name__ == "__main__":
    data = ["a", "b", "c", "d", "e"]
    gen = window_slide_over_list(data, 3)
    for window in gen:
        print(window)
