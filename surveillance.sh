#!/bin/bash

# Définir les variables
INFERENCE_CMD="/NAS/coolio/Barnabe/CODES/neuroharm_kit/NeuroHarm‑Kit_v3/neuroharmo_toolkit.sh"
INPUT_FILE="/NAS/coolio/protocoles/galan/data/bids/sub-00005/ses-20151211/anat/sub-00005_ses-20151211_acq-201T1_3D_TFE3MIN3D_T1w.nii.gz"
OUTPUT_DIR="/NAS/coolio/Barnabe/CODES/test_docker"
OPTIONS="--apply_preproc_steps True --save_preprocess True"
# OPTIONS="--n_axial_slices 200 --save_preprocess True" # pour MURD

SECONDS=0

# Lancer l'inférence en arrière-plan
echo "Lancement de l'inférence..."
$INFERENCE_CMD iguane $INPUT_FILE $OUTPUT_DIR $OPTIONS &
INFERENCE_PID=$!

# Identifier le conteneur Docker associé à l'inférence
CONTAINER_ID=$(docker ps --filter "ancestor=neuroharmo_toolkit" --format "{{.ID}}")

# Fonction pour surveiller l'utilisation du CPU du conteneur
monitor_cpu() {
  echo "Surveillance de l'utilisation du CPU du conteneur Docker..."
  while kill -0 $INFERENCE_PID 2>/dev/null; do
    docker stats --no-stream --format "{{.CPUPerc}}" $CONTAINER_ID >> cpu_usage.txt
    sleep 0.1
  done
}

monitor_mem_usage() {
  echo "Surveillance de l'utilisation de la mémoire du conteneur Docker..."
  while kill -0 $INFERENCE_PID 2>/dev/null; do
    docker stats --no-stream --format "{{.MemUsage}}" $CONTAINER_ID | \
    awk -F' / ' '{print $1}' | \
    sed -E 's/([0-9.]+)(MiB|GiB)/\1 \2/' | \
    while read -r usage unit; do
      if [[ "$unit" == "GiB" ]]; then
        usage=$(echo "$usage * 1024" | bc)
      fi
      echo "$usage" >> mem_usage.txt
    done
    sleep 0.1
  done
}

monitor_gpu() {
  echo "Surveillance de l'utilisation de la mémoire GPU..."
  while kill -0 "$INFERENCE_PID" 2>/dev/null; do
    if nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits &> /dev/null; then
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> gpu_memory_usage.txt
    else
      echo 0 >> gpu_memory_usage.txt
    fi
    sleep 0.1
  done
}

# Démarrer les fonctions de surveillance en arrière-plan
monitor_cpu &
MONITOR_CPU_PID=$!
monitor_mem_usage &
MONITOR_MEM_PID=$!
monitor_gpu &
MONITOR_GPU_PID=$!

# Attendre la fin de l'inférence
wait $INFERENCE_PID

# Terminer les processus de surveillance
kill $MONITOR_CPU_PID
kill $MONITOR_MEM_PID
kill $MONITOR_GPU_PID

# Analyser les résultats
echo "Analyse des résultats..."
echo "Temps d'exécution total : ${SECONDS} secondes"

# Extraire les valeurs maximales
MAX_CPU_USAGE=$(awk '{print $1}' cpu_usage.txt | sort -n | tail -n 1)
MAX_GPU_MEMORY=$(awk '{print $1}' gpu_memory_usage.txt | sort -n | tail -n 1)
MAX_MEM_USAGE=$(awk '{print $1}' mem_usage.txt | sort -n | tail -n 1)

# Afficher les résultats
echo "Temps d'exécution total : $EXEC_TIME"
echo "Utilisation maximale du CPU : $MAX_CPU_USAGE"
echo "Utilisation maximale de la mémoire GPU : ${MAX_GPU_MEMORY}MiB"
echo "Utilisation maximale de la mémoire du conteneur : $MAX_MEM_USAGE"

# Nettoyer les fichiers temporaires
rm cpu_usage.txt gpu_memory_usage.txt mem_usage.txt
