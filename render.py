import tkinter as tk

import numpy as np

from numpy.typing import NDArray
from itertools import chain, zip_longest


def print_bonds_lattice(matrix: NDArray[np.int64], h_bonds: NDArray[np.int64], w_bonds: NDArray[np.int64]) -> None:
  H, W = matrix.shape

  def node_row(i):
    nodes = "".join("●" if matrix[i, j] else "○" for j in range(W))
    links = "".join("─" if h_bonds[i, j] else " " for j in range(W - 1))
    return "".join(a + b for a, b in zip_longest(nodes, links, fillvalue="")).rstrip()

  def link_row(i):
    return "".join("│ " if w_bonds[i, j] else "  " for j in range(W)).rstrip()

  lines = chain.from_iterable([node_row(i)] + ([link_row(i)] if i < H - 1 else []) for i in range(H))

  print("\n".join(lines))


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
