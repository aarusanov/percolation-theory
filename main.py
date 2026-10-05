import numpy as np
import pandas as pd

from tqdm import tqdm
from numpy.typing import NDArray
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from stats import concentration, find_label_clusters, find_percolating_cluster, mean_cluster_size, pearson_test, scott_bins, sigmoid
from gen import random_bond_occupation, random_occupation, random_occupation_continuum
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


class Solution4:

  @classmethod
  def solve_task_1(cls, L: int, p: float, step: float, k: int = 100) -> pd.DataFrame:
    trials = (cls.__run_experiment(L, p) for p in tqdm(np.arange(p, 1.0, step), desc=f'L={L}') for _ in range(k))
    return pd.DataFrame(trials, columns=['p', 'P(p)']).groupby('p').mean()

  @classmethod
  def solve_task_2(cls, sizes: tuple[int, ...], p: float, step: float, k: int = 100) -> None:
    tables = [cls.solve_task_1(L, p, step, k)['P(p)'].rename(L) for L in sizes]
    pd.concat(tables, axis=1).to_csv('out/results.csv', sep='\t', float_format='%.4f')

  @classmethod
  def solve_task_3(cls) -> None:
    df = pd.read_csv('out/results.csv', sep='\t', index_col='p')

    ax = df.plot(marker='o', linestyle='none', xlabel='p', ylabel='P(p)', xlim=(0.525, 0.675))
    xs = np.linspace(df.index.min(), df.index.max(), 300)

    p = df.index.to_numpy()

    fits = [curve_fit(sigmoid, p, P, maxfev=10**6)[0] for _, P in df.items()]
    res = pd.Series([pc for pc, _ in fits], index=df.columns.astype(int), name='pc', dtype=float)

    for i, (pc, a) in enumerate(fits):
        ax.plot(xs, sigmoid(xs, pc, a), color=f'C{i}')

    print(res.to_string())

    res = res.to_frame()
    res['x'] = res.index ** - (3 / 4)

    k, b = np.polyfit(res['x'], res['pc'], 1)

    res.plot(x='x', y='pc', style='o', legend=False)
    plt.plot(res['x'], k * res['x'] + b)
    plt.show()

  @classmethod
  def __run_experiment(cls, L: int, p: float) -> tuple[np.float64, int]:
    matrix = random_occupation((L, L), p)
    labels = find_label_clusters(matrix)
    return p, find_percolating_cluster(labels)[0]


class Solution5:

  @classmethod
  def solve_task_1(cls, L: int, p: float, step: float, k: int = 100) -> pd.DataFrame:
    trials = (cls.__run_experiment(L, p) for p in tqdm(np.arange(p, 1.0, step), desc=f'L={L}') for _ in range(k))
    df = pd.DataFrame(trials, columns=['p', 'S', 'Pinf']).groupby('p').mean()
    df.to_csv('out/task5.csv', sep='\t', float_format='%.4f')

  @classmethod
  def solve_task_2(cls) -> None:
    df = pd.read_csv('out/task5.csv', sep='\t')

    df.plot(x='p', y='S', ylabel='S', legend=True, ylim=(0, df['S'].quantile(.95)))
    df[df['Pinf'] > 0].plot(x='p', y='Pinf', ylabel='P∞', legend=True, xlim=(0, None))
    plt.show()

  @classmethod
  def __run_experiment(cls, L: int, p: float) -> tuple[np.float64, np.float64, np.float64]:
    matrix = random_occupation((L, L), p)
    labels = find_label_clusters(matrix)

    sizes = np.bincount(labels.ravel())[1:]
    found, label, size = find_percolating_cluster(labels)

    if found:
      sizes = np.delete(sizes, label - 1)

    return p, mean_cluster_size(sizes), size / matrix.size


if __name__ == '__main__':
  np.random.seed(42)

  #Solution1.solve_task_1(10, 0.2)
  #Solution1.solve_task_2(1000)

  #Solution2.solve_task_1(10, p_bond=0.5)
  #Solution2.solve_task_2(10, p_site=0.6, p_bond=0.5)
  #Solution2.solve_task_3(100, r=2, p=0.4)

  #Solution3.solve_task_1(10, 0.5)
  #Solution3.solve_task_2(10, 0.5)

  #Solution4.solve_task_2((100, 200, 500), p=0.2, step=.02)
  #Solution4.solve_task_3()

  Solution5.solve_task_1(100, p=0.2, step=.02)
  Solution5.solve_task_2()
