import numpy as np

from numpy.typing import NDArray

from utils import torus_distance


def random_occupation(shape: tuple[int, int], p: float) -> NDArray[np.int64]:
  return (np.random.random(shape) < p).astype(np.int64)


def random_bond_occupation(matrix: NDArray[np.int64], p: np.float64) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
  h_mask = (matrix[:, :-1] == 1) & (matrix[:, 1:] == 1)
  w_mask = (matrix[:-1, :] == 1) & (matrix[1:, :] == 1)

  hbonds = (h_mask & (np.random.random(h_mask.shape) < p)).astype(np.int64)
  vbonds = (w_mask & (np.random.random(w_mask.shape) < p)).astype(np.int64)
  return hbonds, vbonds


def random_occupation_continuum(shape: tuple[int, int], radius: np.float64, p: np.float64) -> NDArray[np.float64]:
  height, width = shape
  centers = []

  while len(centers) * np.pi * radius ** 2 / (height * width) <= p:
    point = np.random.uniform((0, 0), (width, height))

    if all(torus_distance(point, center, (width, height)) >= 2 * radius for center in centers):
      centers.append(point)

  return np.array(centers)
