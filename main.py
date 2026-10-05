import numpy as np
import pandas as pd

from numpy.typing import NDArray

from gen import random_bond_occupation, random_occupation, random_occupation_continuum
from stats import concentration, find_label_clusters, find_percolating_cluster, pearson_test, scott_bins
from render import print_bonds_lattice, print_circles
from utils import save_matrix


class Solution1:

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
  def __run_experiment(cls, L: int, p: float) -> tuple[np.float64, int, np.float64, NDArray[np.int64]]:
    matrix = random_occupation((L, L), p)

    n_bins = scott_bins(matrix)
    mean = concentration(matrix)

    return *pearson_test(matrix, n_bins, p), mean, matrix


class Solution2:

  @classmethod
  def solve_task_1(cls, L: int, p_bond: float) -> None:
    matrix = np.ones((L, L), dtype=np.int64)

    hbonds, vbonds = random_bond_occupation(matrix, p_bond)

    print_bonds_lattice(matrix, hbonds, vbonds)

  @classmethod
  def solve_task_2(cls, L: int, p_site: float, p_bond: float) -> None:
    matrix = random_occupation((L, L), p_site)

    hbonds, vbonds = random_bond_occupation(matrix, p_bond)

    print_bonds_lattice(matrix, hbonds, vbonds)

  @classmethod
  def solve_task_3(cls, L: int, r: float, p: float) -> None:
    shape = (L, L)

    centers = random_occupation_continuum(shape, r, p)

    print_circles(centers, r, shape, scale=8)


class Solution3:

  @classmethod
  def solve_task_1(cls, L: int, p: float) -> None:
    matrix = random_occupation((L, L), p)

    labels = find_label_clusters(matrix)

    sizes = np.bincount(labels.ravel())[1:]
    values, counts = np.unique(sizes, return_counts=True)

    save_matrix(labels, f'out/clusters_p{p}.txt')
    np.savetxt(f'out/cluster_sizes_p{p}.txt', np.column_stack([values, counts]), fmt='%d')

    print(labels)
    print('size count')
    print(np.column_stack([values, counts]))

  @classmethod
  def solve_task_2(cls, L: int, p: float) -> None:
    matrix = random_occupation((L, L), p)

    labels = find_label_clusters(matrix)

    found, label, size = find_percolating_cluster(labels)
    print(labels)
    print(f'percolation = {found} label = {label} size = {size}')


if __name__ == '__main__':
  np.random.seed(42)

  #Solution1.solve_task_1(10, 0.2)
  #Solution1.solve_task_2(1000)

  #Solution2.solve_task_1(10, p_bond=0.5)
  #Solution2.solve_task_2(10, p_site=0.6, p_bond=0.5)
  #Solution2.solve_task_3(100, r=2, p=0.4)

  #Solution3.solve_task_1(10, 0.5)
  #Solution3.solve_task_2(10, 0.5)
