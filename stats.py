import numpy as np

from numpy.typing import NDArray


def concentration(matrix: NDArray[np.int64]) -> np.float64:
  return np.count_nonzero(matrix) / matrix.size


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


def find_label_clusters(matrix: NDArray[np.int64]) -> NDArray[np.int64]:
  labels = np.zeros(matrix.shape, dtype=np.int64)
  parent = [0]

  def find(x: int) -> int:
    while parent[x] != x:
      x = parent[x]
    return x

  for i, j in zip(*np.nonzero(matrix)):
    up = find(labels[i - 1, j]) if i else 0
    left = find(labels[i, j - 1]) if j else 0

    if up and left:
      parent[max(up, left)] = min(up, left)

    if up or left:
      labels[i, j] = max(up, left)
    else:
      labels[i, j] = len(parent)
      parent.append(len(parent))

  occupied = labels > 0
  labels[occupied] = [find(x) for x in labels[occupied]]
  labels[occupied] = np.unique(labels[occupied], return_inverse=True)[1] + 1

  return labels


def find_percolating_cluster(labels: NDArray[np.int64]) -> tuple[int, int, int]:
  spanning = np.intersect1d(labels[:, 0], labels[:, -1])
  spanning = spanning[spanning > 0]

  if not spanning.size:
    return 0, 0, 0

  sizes = np.bincount(labels.ravel())
  label = int(spanning[np.argmax(sizes[spanning])])

  return 1, label, int(sizes[label])


def sigmoid(p: NDArray[np.float64], pc: float, a: float) -> NDArray[np.float64]:
  return (1 + np.exp(-(p - pc) * a)) ** -1
