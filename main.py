import numpy as np
import pandas as pd

from numpy.typing import NDArray


def save_matrix(matrix: NDArray[np.int64], filename: str) -> None:
  np.savetxt(filename, matrix, fmt='%d')


def concentration(matrix: NDArray[np.int64]) -> np.float64:
  return np.count_nonzero(matrix) / matrix.size


def random_occupation(matrix: NDArray[np.int64], p: np.float64) -> None:
  while concentration(matrix) < p:
    free_flat = np.flatnonzero(matrix == 0)
    if free_flat.size == 0:
      break
    matrix.flat[np.random.choice(free_flat)] = 1


def scott_bins(matrix: NDArray[np.int64]) -> int:
  opt = int(np.power(2 * matrix.size / 3, 1/9))
  return max(1, (opt + 1) ** 3)


def pearson_test(matrix: NDArray[np.int64], n_bins: int, expected_prob: np.float64) -> tuple[np.float64, int]:
  n_intervals = int(np.sqrt(n_bins))

  rows, cols = matrix.shape
  height, width = rows // n_intervals, cols // n_intervals

  h_adj = height * n_intervals
  w_adj = width * n_intervals
  matrix_adj = matrix[:h_adj, :w_adj]

  blocks = matrix_adj.reshape(n_intervals * n_intervals, height, width)
  observed = np.count_nonzero(blocks, axis=(1, 2))

  expected = np.full(n_intervals * n_intervals, height * width * expected_prob)

  return np.sum((observed - expected) ** 2 / expected), n_intervals * n_intervals - 1


class Solution:

  @classmethod
  def solve_task_1(cls, L: int, p: float) -> None:
    _, _, mean, matrix = cls.__run_experiment(L, p)

    print(f'p = {p} mean = {mean}')
    print(matrix)

  @classmethod
  def solve_task_2(cls, L: int) -> None:
    for p in np.arange(0.2, 1.0, 0.1):
      data = [cls.__run_experiment(L, p) for _ in range(100)]

      df = pd.DataFrame(data, columns=['chi2', 'df', 'concentration', 'bins'])

      print(f"p = {p:.1f}")
      print(df[['chi2', 'df', 'concentration']].mean().round(4))
      print(df['concentration'].head().values)
      print()

  @classmethod
  def __run_experiment(cls, L: int, p: float) -> tuple[np.float64, int, int]:
    shape = (L, L)
    matrix = np.zeros(shape, dtype=int)

    random_occupation(matrix, p)

    n_bins = scott_bins(matrix)
    mean = concentration(matrix)

    return *pearson_test(matrix, n_bins, p), mean, matrix


if __name__ == '__main__':
  np.random.seed(42)

  Solution.solve_task_1(10, 0.2)
  Solution.solve_task_2(100)
