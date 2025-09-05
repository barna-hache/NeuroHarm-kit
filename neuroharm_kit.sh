#!/usr/bin/env bash
# Wrapper simplifié pour neuroharmo_toolkit

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <algo> <input_image> <output_dir> [--options]"
  exit 1
fi

ALGO="$1"; INPUT="$2"; OUTPUT_DIR="$3"; shift 3
IMAGE="neuroharmo_kit:latest"

# Résoudre les chemins absolus
INPUT_ABS=$(readlink -f "$INPUT")
OUTPUT_ABS=$(readlink -f "$OUTPUT_DIR")

# Détection du GPU
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
  echo "GPU détecté — exécution avec support GPU"
  GPU_OPTS="--gpus all --privileged"
else
  echo "Pas de GPU NVIDIA détecté — exécution CPU seulement"
  GPU_OPTS="--privileged"
fi

# Lancer le conteneur Docker
docker run --rm $GPU_OPTS \
  -v "${INPUT_ABS%/*}:/data/in" \
  -v "${OUTPUT_ABS}:/data/out" \
  ${IMAGE} \
  $ALGO "/data/in/$(basename "$INPUT_ABS")" --output_dir /data/out "$@"



# #!/usr/bin/env bash
# # Wrapper simplifié pour neuroharmo_toolkit

# if [ "$#" -lt 3 ]; then
#   echo "Usage: $0 <algo> <input_image> <output_dir> [--options]"
#   exit 1
# fi

# ALGO="$1"; INPUT="$2"; OUTPUT_DIR="$3"; shift 3
# IMAGE="neuroharmo_kit:latest"

# # Resolve absolute paths
# INPUT_ABS=$(readlink -f "$INPUT")
# OUTPUT_ABS=$(readlink -f "$OUTPUT_DIR")


#  #--gpus all \
# docker run --rm \
#   --gpus all --privileged \
#   -v "${INPUT_ABS%/*}:/data/in" \
#   -v "${OUTPUT_ABS}:/data/out" \
#   ${IMAGE} \
#   $ALGO \
#     "/data/in/$(basename "$INPUT_ABS")" \
#     --output_dir /data/out \
#     "$@"