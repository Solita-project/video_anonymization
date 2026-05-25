#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
FORCE_RECREATE="${FORCE_RECREATE:-0}"

CORE_VENV="venvs/core"
SPEAKER_VENV="venvs/speaker"
TRANSCRIPT_VENV="venvs/transcript"
TTS_VENV="venvs/tts"
VIDEO_VENV="venvs/video"

CORE_TORCH_PACKAGES="torch"
FULL_TORCH_PACKAGES="torch torchvision torchaudio"

print_header() {
    printf "\n============================================================\n"
    printf "%s\n" "$1"
    printf "============================================================\n"
}

check_python_version() {
    "${PYTHON_BIN}" - <<'PY'
import sys
required = (3, 10)
current = sys.version_info[:2]
print(f"Using Python {sys.version.split()[0]}")
if current != required:
    raise SystemExit(
        "ERROR: Python 3.10 is required for this project.\n"
        f"Current Python is {sys.version.split()[0]}.\n\n"
        "Install Python 3.10 and run again, or set PYTHON_BIN to the correct executable."
    )
PY
}

venv_python() {
    local venv_dir="$1"
    if [[ "$OS_KIND" == "windows" ]]; then
        printf "%s/Scripts/python.exe" "$venv_dir"
    else
        printf "%s/bin/python" "$venv_dir"
    fi
}

create_venv() {
    local venv_dir="$1"

    if [[ "${FORCE_RECREATE}" == "1" && -d "$venv_dir" ]]; then
        print_header "Removing existing venv: ${venv_dir}"
        rm -rf "$venv_dir"
    fi

    if [[ ! -f "$(venv_python "$venv_dir")" ]]; then
        print_header "Creating venv: ${venv_dir}"
        "${PYTHON_BIN}" -m venv "$venv_dir"
    else
        print_header "Venv already exists: ${venv_dir}"
    fi
}

upgrade_base_tools() {
    local python_exe="$1"
    "${python_exe}" -m pip install --upgrade "pip<27" "setuptools<82" wheel packaging
}

install_requirements() {
    local python_exe="$1"
    local requirements_file="$2"

    if [[ ! -f "$requirements_file" ]]; then
        printf "\nERROR: Requirements file not found: %s\n" "$requirements_file"
        exit 1
    fi

    print_header "Installing requirements: ${requirements_file}"
    "${python_exe}" -m pip install -r "$requirements_file"
    "${python_exe}" -m pip install --upgrade "setuptools<82"
}

install_spacy_finnish_model() {
    local python_exe="$1"
    print_header "Installing spaCy Finnish model: fi_core_news_sm"
    "${python_exe}" -m spacy download fi_core_news_sm
}

PYTORCH_CPU_INDEX="${PYTORCH_CPU_INDEX:-https://download.pytorch.org/whl/cpu}"

install_cpu_torch() {
    local python_exe="$1"
    local packages="$2"
    local env_name="$3"

    print_header "Installing CPU PyTorch packages for ${env_name}: ${packages}"
    "${python_exe}" -m pip install ${packages} --index-url "${PYTORCH_CPU_INDEX}"
    "${python_exe}" -m pip install --upgrade "setuptools<82"
}

verify_cpu_torch() {
    local python_exe="$1"
    local env_name="$2"
    local check_vision="$3"
    local check_audio="$4"

    print_header "Verifying CPU torch stack in ${env_name}"

    "${python_exe}" - <<PY
import torch
check_vision = "${check_vision}" == "1"
check_audio = "${check_audio}" == "1"
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if check_vision:
    import torchvision
    from torchvision.ops import nms
    print("torchvision:", torchvision.__version__)
    print("torchvision nms: OK")
if check_audio:
    import torchaudio
    print("torchaudio:", torchaudio.__version__)
print("${env_name}: OK")
PY
}

setup_env_cpu() {
    local env_name="$1"
    local venv_dir="$2"
    local requirements_file="$3"
    local torch_packages="$4"
    local check_vision="$5"
    local check_audio="$6"
    local install_spacy="$7"

    create_venv "$venv_dir"
    local python_exe
    python_exe="$(venv_python "$venv_dir")"

    upgrade_base_tools "$python_exe"
    install_cpu_torch "$python_exe" "$torch_packages" "$env_name"
    install_requirements "$python_exe" "$requirements_file"

    if [[ "$install_spacy" == "1" ]]; then
        install_spacy_finnish_model "$python_exe"
    fi

    install_cpu_torch "$python_exe" "$torch_packages" "$env_name"
    verify_cpu_torch "$python_exe" "$env_name" "$check_vision" "$check_audio"
}

OS_KIND="windows"

main() {
    check_python_version

    printf "\nUsing PyTorch CPU wheel index: ${PYTORCH_CPU_INDEX}\n"

    setup_env_cpu "core"       "$CORE_VENV"       "requirements/core/base.txt"       "$CORE_TORCH_PACKAGES" "0" "0" "0"
    setup_env_cpu "speaker"    "$SPEAKER_VENV"    "requirements/speaker/cpu.txt" "$FULL_TORCH_PACKAGES" "1" "1" "0"
    setup_env_cpu "transcript" "$TRANSCRIPT_VENV" "requirements/transcript/cpu.txt" "$FULL_TORCH_PACKAGES" "1" "1" "1"
    setup_env_cpu "tts"        "$TTS_VENV"        "requirements/tts/cpu.txt" "$FULL_TORCH_PACKAGES" "1" "1" "0"
    setup_env_cpu "video"      "$VIDEO_VENV"      "requirements/video/cpu.txt" "$FULL_TORCH_PACKAGES" "1" "1" "0"

    print_header "CPU setup complete"
    printf "Activate core venv with:\n"
    printf "  source venvs/core/Scripts/activate\n\n"
    printf "Run pipeline with:\n"
    printf "  python run/run_pipeline.py\n"
}

main "$@"
