import numpy as np
import pandas as pd
import tkinter as tk

from numpy.typing import NDArray
from itertools import chain, zip_longest


def save_matrix(matrix: NDArray[np.int64], filename: str) -> None:
  np.savetxt(filename, matrix, fmt='%d')


def concentration(matrix: NDArray[np.int64]) -> np.float64:
  return np.count_nonzero(matrix) / matrix.size


def random_occupation(matrix: NDArray[np.int64], p: np.float64) -> None:
  matrix[:] = np.random.random(matrix.shape) < p


def random_bond_occupation(matrix: NDArray[np.int64], p: np.float64) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
  h_mask = (matrix[:, :-1] == 1) & (matrix[:, 1:] == 1)
  w_mask = (matrix[:-1, :] == 1) & (matrix[1:, :] == 1)

  hbonds = (h_mask & (np.random.random(h_mask.shape) < p)).astype(np.int64)
  vbonds = (w_mask & (np.random.random(w_mask.shape) < p)).astype(np.int64)
  return hbonds, vbonds


def circle_area(radius: np.float64) -> np.float64:
  return np.pi * radius ** 2


def torus_distance(a: NDArray[np.float64], b: NDArray[np.float64], shape: tuple[int, int]) -> np.float64:
  height, width = shape

  dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
  dx, dy= min(dx, width - dx), min(dy, height - dy)

  return np.hypot(dx, dy)


def random_occupation_continuum(shape: tuple[int, int], radius: np.float64, p: np.float64) -> NDArray[np.float64]:
  height, width = shape
  centers = []

  while len(centers) * circle_area(radius) / (height * width) <= p:
    point = np.random.uniform((0, 0), (width, height))

    if all(torus_distance(point, center, (width, height)) >= 2 * radius for center in centers):
      centers.append(point)

  return np.array(centers)


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


def print_circles(centers: NDArray[np.float64], radius: np.float64, shape: tuple[int, int], scale: int = 6) -> None:
  height, width = shape

  root = tk.Tk()

  canvas = tk.Canvas(root, width=width*scale, height=height*scale, bg="white")
  canvas.pack()

  def visible_images(x, y):
    return (
      (px, py)
      for dx in (-width, 0, width)
      for dy in (-height, 0, height)
      for px, py in [(x + dx, y + dy)]
      if -radius <= px <= width + radius and -radius <= py <= height + radius
    )

  def oval_box(px, py):
    cx, cy = px * scale, (height - py) * scale
    r = radius * scale
    return cx-r, cy-r, cx+r, cy+r

  for px, py in chain.from_iterable(visible_images(x, y) for x, y in centers):
    canvas.create_oval(*oval_box(px, py), outline="blue", width=1)

  root.mainloop()


class Solution:

  @classmethod
  def solve_task_1(cls, L: int, p: float) -> None:
    _, _, mean, matrix = cls.__run_experiment(L, p)

    print(f'p = {p} mean = {mean}')
    print(matrix)

  @classmethod
  def solve_task_2(cls, L: int) -> None:
    for p in np.arange(0.2, 1.0, 0.1):
      chi2, dof, mean, _ = cls.__run_experiment(L, p)

      data = [cls.__run_experiment(L, p) for _ in range(100)]

      df = pd.DataFrame(data, columns=['chi2', 'df', 'concentration', 'bins'])

      mean_df = df[['chi2', 'df', 'concentration']].mean()

      print(f"p = {p:.1f}")
      print(f"1:   chi2 = {chi2:.4f} df = {dof} concentration = {mean:.4f}")
      print(f"100: chi2 = {mean_df['chi2']:.4f} df = {mean_df['df']:.0f} concentration = {mean_df['concentration']:.4f}")
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
  def solve_task_5(cls, L: int, r: float, p: float) -> None:
    shape = (L, L)

    centers = random_occupation_continuum(shape, r, p)

    print_circles(centers, r, shape, scale=8)

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

  #Solution.solve_task_1(10, 0.2)
  #Solution.solve_task_2(1000)
  #Solution.solve_task_3(10, p_bond=0.2)
  #Solution.solve_task_4(10, p_site=0.5, p_bond=0.3)
  Solution.solve_task_5(100, r=2, p=0.4)
