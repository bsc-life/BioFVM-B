#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_DIR="${1:-${REPO_ROOT}/output/density_diffusion}"
ITERATIONS="${ITERATIONS:-500}"
DT="${DT:-0.01}"
PLOT_KIND="${PLOT_KIND:-contour}"
PLANE="${PLANE:-xy}"

mkdir -p "${OUTPUT_DIR}"

cd "${REPO_ROOT}"
make density_diffusion_demo

./examples/density_diffusion_demo "${OUTPUT_DIR}" "${ITERATIONS}" "${DT}"

PLOT_DIR="${OUTPUT_DIR}/plots"
mkdir -p "${PLOT_DIR}"

PLOT_COMMAND="addpath('${REPO_ROOT}/matlab'); plot_density_diffusion('${OUTPUT_DIR}/density_diffusion_initial.mat', '${PLOT_DIR}', '${PLOT_KIND}', '${PLANE}'); plot_density_diffusion('${OUTPUT_DIR}/density_diffusion_final.mat', '${PLOT_DIR}', '${PLOT_KIND}', '${PLANE}');"

if [[ "${SKIP_PLOTS:-0}" == "1" ]]; then
	echo "SKIP_PLOTS=1; skipped PNG plot generation."
elif command -v octave >/dev/null 2>&1; then
	if ! octave --quiet --eval "${PLOT_COMMAND}"; then
		echo "Octave failed; MAT files were still generated."
	fi
elif python3 -c "import scipy.io, matplotlib" >/dev/null 2>&1; then
	export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-biofvm}"
	python3 "${REPO_ROOT}/scripts/plot_density_diffusion.py" \
		"${OUTPUT_DIR}/density_diffusion_initial.mat" "${PLOT_DIR}" \
		--kind "${PLOT_KIND}" --plane "${PLANE}"
	python3 "${REPO_ROOT}/scripts/plot_density_diffusion.py" \
		"${OUTPUT_DIR}/density_diffusion_final.mat" "${PLOT_DIR}" \
		--kind "${PLOT_KIND}" --plane "${PLANE}"
else
	echo "No octave or python scipy/matplotlib plotting backend found; skipped PNG plot generation."
	echo "Install Octave to run the MATLAB plotting script without using MATLAB, then run:"
	echo "  sudo apt install octave"
	echo "  ./scripts/run_density_diffusion_plot.sh"
	echo "Or open Octave and run:"
	echo "  addpath('${REPO_ROOT}/matlab')"
	echo "  plot_density_diffusion('${OUTPUT_DIR}/density_diffusion_final.mat', '${PLOT_DIR}', '${PLOT_KIND}', '${PLANE}')"
fi

echo "MAT files:"
echo "  ${OUTPUT_DIR}/density_diffusion_initial.mat"
echo "  ${OUTPUT_DIR}/density_diffusion_final.mat"
echo "Plot directory:"
echo "  ${PLOT_DIR}"
