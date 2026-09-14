import numpy as np
import pandas as pd

from numpy.typing import NDArray
from itertools import chain, zip_longest


def save_matrix(matrix: NDArray[np.int64], filename: str) -> None:
  np.savetxt(filename, matrix, fmt='%d')


def concentration(matrix: NDArray[np.int64]) -> np.float64:
  return np.count_nonzero(matrix) / matrix.size


def random_occupation(matrix: NDArray[np.int64], p: np.float64) -> None:
  height, width = matrix.shape
  while concentration(matrix) <= p:
    x, y = np.random.randint(height), np.random.randint(width)
    matrix[x, y] = 1


def random_bond_occupation(matrix: NDArray[np.int64], p: np.float64) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
  H, W = matrix.shape
  
  h_possible = (matrix[:, :-1] == 1) & (matrix[:, 1:] == 1)
  w_possible = (matrix[:-1, :] == 1) & (matrix[1:, :] == 1)

  n_h = H * (W - 1)
  n_w = (H - 1) * W 

  possible = np.concatenate([np.flatnonzero(h_possible), n_h + np.flatnonzero(w_possible)])

  bonds = np.zeros(n_h + n_w, dtype=np.int64)

  while np.count_nonzero(bonds[possible]) / possible.size <= p:
    bond_idx = possible[np.random.randint(len(possible))]
    bonds[bond_idx] = 1

  return bonds[:n_h].reshape(H, W - 1), bonds[n_h:].reshape(H - 1, W)


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


def format_bonds_lattice(matrix: NDArray[np.int64], h_bonds: NDArray[np.int64],w_bonds: NDArray[np.int64]) -> str:
    H, W = matrix.shape

    def node_row(i):
        nodes = "".join("●" if matrix[i, j] else "○" for j in range(W))
        links = "".join("─" if h_bonds[i, j] else " " for j in range(W - 1))
        return "".join(a + b for a, b in zip_longest(nodes, links, fillvalue="")).rstrip()

    def link_row(i):
        return "".join("│ " if w_bonds[i, j] else "  " for j in range(W)).rstrip()

    lines = chain.from_iterable([node_row(i)] + ([link_row(i)] if i < H - 1 else []) for i in range(H))

    return "\n".join(lines)


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
  def solve_task_3(cls, L: int, p_bond: float) -> None:
    matrix = np.ones((L, L), dtype=np.int64)

    hbonds, vbonds = random_bond_occupation(matrix, p_bond)

    print(format_bonds_lattice(matrix, hbonds, vbonds))

  @classmethod
  def solve_task_4(cls, L: int, p_site: float, p_bond: float) -> None:
    matrix = np.zeros((L, L), dtype=np.int64)

    random_occupation(matrix, p_site)
    hbonds, vbonds = random_bond_occupation(matrix, p_bond)

    print(format_bonds_lattice(matrix, hbonds, vbonds))

  @classmethod
  def __run_experiment(cls, L: int, p: float) -> tuple[np.float64, int, int]:
    shape = (L, L)
    matrix = np.zeros(shape, dtype=int)

    random_occupation(matrix, p)

    n_bins = scott_bins(matrix)
    mean = concentration(matrix)

    return *pearson_test(matrix, n_bins, p), mean, matrix


if __name__ == '__main__':
  np.random.seed()

  #Solution.solve_task_1(10, 0.2)
  #Solution.solve_task_2(100)
  #Solution.solve_task_3(10, p_bond=0.2)
  Solution.solve_task_4(10, p_site=0.9, p_bond=0.3)