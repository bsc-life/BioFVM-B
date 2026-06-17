#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/density_diffusion}"
PLOT_DIR="${PLOT_DIR:-${OUTPUT_DIR}/plots}"
PLOT_KIND="${PLOT_KIND:-contour}"
PLANE="${PLANE:-xy}"

if command -v octave-cli >/dev/null 2>&1; then
	OCTAVE_BIN="octave-cli"
elif command -v octave >/dev/null 2>&1; then
	OCTAVE_BIN="octave"
else
	echo "Octave is required for this script, but it was not found in PATH."
	exit 1
fi

mkdir -p "${PLOT_DIR}"

if [[ "$#" -gt 0 ]]; then
	MAT_FILES=("$@")
else
	MAT_FILES=(
		"${OUTPUT_DIR}/density_diffusion_initial.mat"
		"${OUTPUT_DIR}/density_diffusion_final.mat"
	)
fi

for mat_file in "${MAT_FILES[@]}"; do
	if [[ ! -f "${mat_file}" ]]; then
		echo "Missing MAT file: ${mat_file}"
		exit 1
	fi
done

OCTAVE_SCRIPT="$(mktemp /tmp/biofvm_octave_plot_XXXXXX.m)"
cleanup() {
	rm -f "${OCTAVE_SCRIPT}"
}
trap cleanup EXIT

{
	printf "warning('off', 'all');\n"
	printf "addpath('%s');\n" "${REPO_ROOT}/matlab"
	printf "plot_dir = '%s';\n" "${PLOT_DIR}"
	printf "plot_kind = '%s';\n" "${PLOT_KIND}"
	printf "plane = '%s';\n" "${PLANE}"
	for mat_file in "${MAT_FILES[@]}"; do
		printf "plot_density_diffusion('%s', plot_dir, plot_kind, plane);\n" "${mat_file}"
	done
} > "${OCTAVE_SCRIPT}"

"${OCTAVE_BIN}" --quiet "${OCTAVE_SCRIPT}" \
	2> >(grep -vF "error: ignoring const execution_exception& while preparing to exit" >&2)

echo "Plot directory:"
echo "  ${PLOT_DIR}"
