#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

clean_path() {
    local value="$1"
    value="${value%\"}"; value="${value#\"}"
    value="${value%\'}"; value="${value#\'}"
    value="${value/#\~/$HOME}"
    printf '%s' "$value"
}

yes_no() {
    local answer="$1"
    answer="${answer,,}"
    [[ "$answer" == "y" || "$answer" == "yes" ]]
}

read -r -p "Original image path: " IMAGE_INPUT
IMAGE="$(clean_path "$IMAGE_INPUT")"
if [[ ! -f "$IMAGE" ]]; then
    echo "Error: image not found: $IMAGE"
    exit 1
fi

read -r -p "Crop ratio, e.g. 0.4 for 40% (default 0.4): " CROP
CROP="${CROP:-0.4}"
if ! [[ "$CROP" =~ ^([0-9]+(\.[0-9]+)?|\.[0-9]+)$ ]]; then
    echo "Error: crop ratio must be a number"
    exit 1
fi
if ! python3 -c "import sys; sys.exit(0 if 0 < float(sys.argv[1]) <= 1 else 1)" "$CROP"; then
    echo "Error: crop ratio must be between 0 and 1"
    exit 1
fi

read -r -p "Grayscale image? [y/N]: " GRAY_INPUT
GRAY=no
if yes_no "$GRAY_INPUT"; then
    GRAY=yes
fi

read -r -p "Compare traditional vs improved? [Y/n]: " COMPARE_INPUT
COMPARE_INPUT="${COMPARE_INPUT:-y}"
COMPARE=no
if yes_no "$COMPARE_INPUT"; then
    COMPARE=yes
fi

METHOD=improved
CUSTOM=
if [[ "$COMPARE" == "no" ]]; then
    read -r -p "Which method? type improved or traditional (default improved): " METHOD_INPUT
    METHOD_INPUT="${METHOD_INPUT:-improved}"
    METHOD_INPUT="${METHOD_INPUT,,}"
    if [[ "$METHOD_INPUT" != "improved" && "$METHOD_INPUT" != "traditional" ]]; then
        echo "Error: choose improved or traditional"
        exit 1
    fi
    METHOD="$METHOD_INPUT"
fi

if [[ "$COMPARE" == "yes" || "$METHOD" == "improved" ]]; then
    read -r -p "Use your own cropped 16-bit watermarked image? [y/N]: " CUSTOM_INPUT
    if yes_no "$CUSTOM_INPUT"; then
        read -r -p "Cropped image path: " CUSTOM_PATH_INPUT
        CUSTOM="$(clean_path "$CUSTOM_PATH_INPUT")"
        if [[ ! -f "$CUSTOM" ]]; then
            echo "Error: cropped image not found: $CUSTOM"
            exit 1
        fi
    fi
fi

read -r -p "Random seed (default 2025): " SEED
SEED="${SEED:-2025}"
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "Error: seed must be an integer"
    exit 1
fi

read -r -p "Output folder (default comparison_output): " OUTPUT_INPUT
if [[ "$COMPARE" == "yes" ]]; then
    OUTPUT_INPUT="${OUTPUT_INPUT:-comparison_output}"
else
    OUTPUT_INPUT="${OUTPUT_INPUT:-recovery_output}"
fi
OUTPUT="$(clean_path "$OUTPUT_INPUT")"

ARGS=(--image "$IMAGE" --crop "$CROP" --seed "$SEED" --output "$OUTPUT")
if [[ "$GRAY" == "yes" ]]; then
    ARGS+=(--grayscale)
fi
if [[ -n "$CUSTOM" ]]; then
    ARGS+=(--attacked "$CUSTOM")
fi

echo
if [[ "$COMPARE" == "yes" ]]; then
    echo "Running traditional + improved comparison..."
    python3 "$DIR/run_comparison.py" "${ARGS[@]}"
elif [[ "$METHOD" == "improved" ]]; then
    echo "Running improved method..."
    python3 "$DIR/self_embedded_recovery.py" "${ARGS[@]}"
else
    echo "Running traditional method..."
    python3 "$DIR/baseline_traditional.py" "${ARGS[@]}"
fi
