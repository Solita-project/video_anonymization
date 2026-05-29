# Automatically selects the correct setup script.
# Detects Windows or Linux and checks if NVIDIA GPU is available.
# Usage:
# chmod +x starter/setup_auto.sh
# ./starter/setup_auto.sh

#!/usr/bin/env bash
set -e

# Move to project root
cd "$(dirname "$0")/.."

# Detect operating system
OS_NAME="$(uname -s)"

# Detect NVIDIA GPU
HAS_GPU=false

if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi -L >/dev/null 2>&1; then
        HAS_GPU=true
    fi
fi

# Select setup script
if [[ "$OS_NAME" == MINGW* || "$OS_NAME" == MSYS* || "$OS_NAME" == CYGWIN* ]]; then

    if [[ "$HAS_GPU" == true ]]; then
        SETUP_SCRIPT="starter/setup_windows_gpu.sh"
    else
        SETUP_SCRIPT="starter/setup_windows_cpu.sh"
    fi

elif [[ "$OS_NAME" == Linux* ]]; then

    if [[ "$HAS_GPU" == true ]]; then
        SETUP_SCRIPT="starter/setup_linux_gpu.sh"
    else
        SETUP_SCRIPT="starter/setup_linux_cpu.sh"
    fi

else
    echo "Unsupported operating system: $OS_NAME"
    exit 1
fi

# Show selected setup script
echo "Selected setup script: $SETUP_SCRIPT"

# Run selected setup script
bash "$SETUP_SCRIPT"