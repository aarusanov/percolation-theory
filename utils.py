import numpy as np

from numpy.typing import NDArray


def save_matrix(matrix: NDArray[np.int64], filename: str) -> None:
  np.savetxt(filename, matrix, fmt=f'%{len(str(matrix.max()))}d')


def torus_distance(a: NDArray[np.float64], b: NDArray[np.float64], shape: tuple[int, int]) -> np.float64:
  width, height = shape

  dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
  dx, dy= min(dx, width - dx), min(dy, height - dy)

  return np.hypot(dx, dy)
