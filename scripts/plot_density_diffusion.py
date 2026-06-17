#!/usr/bin/env python3
"""Plot BioFVM multiscale_microenvironment MAT files.

This is a non-MATLAB fallback for scripts/run_density_diffusion_plot.sh.
"""

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io


def plane_axes(plane):
    labels = ("x", "y", "z")
    lookup = {
        "xy": (0, 1),
        "xz": (0, 2),
        "yz": (1, 2),
    }
    if plane not in lookup:
        raise ValueError(f'Unknown plane "{plane}". Use xy, xz, or yz.')
    plotted = lookup[plane]
    sliced = next(axis for axis in range(3) if axis not in plotted)
    return plotted, sliced, labels


def middle_slice(matrix, plane):
    plotted, sliced, labels = plane_axes(plane)
    slice_values = np.unique(matrix[sliced, :])
    slice_value = slice_values[(len(slice_values) - 1) // 2]
    section = matrix[:, matrix[sliced, :] == slice_value]
    return section, plotted, sliced, labels, slice_value


def density_grid(section, plotted, density_row):
    axis_a = np.unique(section[plotted[0], :])
    axis_b = np.unique(section[plotted[1], :])

    values = np.full((len(axis_b), len(axis_a)), np.nan)
    a_index = {value: index for index, value in enumerate(axis_a)}
    b_index = {value: index for index, value in enumerate(axis_b)}

    for col in range(section.shape[1]):
        i = a_index[section[plotted[0], col]]
        j = b_index[section[plotted[1], col]]
        values[j, i] = section[density_row, col]

    return axis_a, axis_b, values


def plot_file(file_name, output_dir, plot_kind, plane):
    loaded = scipy.io.loadmat(file_name)
    matrix = loaded["multiscale_microenvironment"]
    section, plotted, sliced, labels, slice_value = middle_slice(matrix, plane)

    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(file_name).stem

    for density_row in range(4, section.shape[0]):
        density_index = density_row - 3
        axis_a, axis_b, values = density_grid(section, plotted, density_row)

        fig = plt.figure(figsize=(6.5, 5.4), dpi=200)
        actual_kind = plot_kind
        if plot_kind == "surface":
            try:
                ax = fig.add_subplot(111, projection="3d")
                aa, bb = np.meshgrid(axis_a, axis_b)
                image = ax.plot_surface(aa, bb, values, cmap="viridis", linewidth=0)
                ax.set_zlabel("concentration")
            except ValueError:
                print("Matplotlib 3D projection is unavailable; writing contour plot instead.")
                fig.clear()
                ax = fig.add_subplot(111)
                image = ax.contourf(axis_a, axis_b, values, levels=30, cmap="viridis")
                ax.set_aspect("equal", adjustable="box")
                actual_kind = "contour"
        else:
            ax = fig.add_subplot(111)
            image = ax.contourf(axis_a, axis_b, values, levels=30, cmap="viridis")
            ax.set_aspect("equal", adjustable="box")

        ax.set_title(
            f"{base_name} density {density_index} {plane.upper()} slice at "
            f"{labels[sliced]} = {slice_value:g}"
        )
        ax.set_xlabel(f"{labels[plotted[0]]} (um)")
        ax.set_ylabel(f"{labels[plotted[1]]} (um)")
        fig.colorbar(image, ax=ax, label="concentration")
        fig.tight_layout()

        output_file = output_dir / f"{base_name}_density_{density_index}_{plane}_{actual_kind}.png"
        fig.savefig(output_file)
        plt.close(fig)
        print(f"Wrote {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name")
    parser.add_argument("output_dir")
    parser.add_argument("--kind", choices=("contour", "surface"), default="contour")
    parser.add_argument("--plane", choices=("xy", "xz", "yz"), default="xy")
    args = parser.parse_args()

    plot_file(Path(args.file_name), Path(args.output_dir), args.kind, args.plane)


if __name__ == "__main__":
    main()
