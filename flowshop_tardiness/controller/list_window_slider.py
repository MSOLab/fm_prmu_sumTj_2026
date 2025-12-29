from collections import deque
from itertools import islice
from typing import Any, Iterator, Sequence


def window_slide_over_list(
    iterable: Sequence[Any], n: int
) -> Iterator[tuple[Any, ...]]:
    """Generate a sliding window of width n over data from the iterable.

    Args:
        iterable (Sequence[Any]): input sequence
        n (int): window size

    Raises:
        ValueError: If the window size n is not positive.
        ValueError: If the window size n is greater than the length of the iterable.

    Yields:
        tuple[Any, ...]: tuples of length n, each representing the current window
    """
    if n <= 0:
        raise ValueError("Window size n must be positive.")
    if n > len(iterable):
        raise ValueError(
            "Window size n must not be greater than the length of the iterable."
        )

    # Use iter() so input can be any iterable (streams too)
    it = iter(iterable)
    window = deque(islice(it, n), maxlen=n)
    if len(window) == n:
        yield tuple(window)
    for elem in it:
        window.append(elem)  # deque drops leftmost on overflow and appends right (O(1))
        yield tuple(window)


if __name__ == "__main__":
    data = ["a", "b", "c", "d", "e"]
    gen = window_slide_over_list(data, 3)
    for window in gen:
        print(window)
