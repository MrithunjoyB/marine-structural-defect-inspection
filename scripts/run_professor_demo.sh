#!/bin/sh
set -eu

usage() {
  echo "Usage: scripts/run_professor_demo.sh --input IMAGE [--output-base DIR] [--venv DIR] [--open]" >&2
  exit 2
}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd -P)
input_path=
output_base="$repo_root/build/professor-demo-runs"
venv_path=
open_after=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --input)
      [ "$#" -ge 2 ] || usage
      input_path=$2
      shift 2
      ;;
    --output-base)
      [ "$#" -ge 2 ] || usage
      output_base=$2
      shift 2
      ;;
    --venv)
      [ "$#" -ge 2 ] || usage
      venv_path=$2
      shift 2
      ;;
    --open)
      open_after=1
      shift
      ;;
    *)
      usage
      ;;
  esac
done

[ -n "$input_path" ] || usage
[ -f "$input_path" ] || {
  echo "Input file not found: $input_path" >&2
  exit 3
}

if [ -n "$venv_path" ]; then
  [ -f "$venv_path/bin/activate" ] || {
    echo "Virtual environment activation script not found: $venv_path/bin/activate" >&2
    exit 3
  }
  # shellcheck disable=SC1090
  . "$venv_path/bin/activate"
fi

command -v structvision-professor-demo >/dev/null 2>&1 || {
  echo "structvision-professor-demo is not installed in the selected environment." >&2
  echo "Activate the prepared environment; this launcher never installs packages." >&2
  exit 4
}

mkdir -p -- "$output_base"
timestamp=$(date -u '+%Y%m%dT%H%M%SZ')
output_dir="$output_base/demo-run-$timestamp"

structvision-professor-demo \
  --input "$input_path" \
  --output-dir "$output_dir"

echo
echo "Completed output folder: $output_dir"

if [ "$open_after" -eq 1 ]; then
  if [ "$(uname -s)" = "Darwin" ]; then
    open "$output_dir/OUTPUT/overlay.png"
    open "$output_dir"
  else
    echo "--open is available only on macOS; output was created successfully."
  fi
fi
