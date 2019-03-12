#copy images and xmls into images folder (into subfolder train and test)

BASE_DIR="$(cd "$(dirname "$0")"; pwd)";
echo "BASE_DIR => $BASE_DIR";
MODELS_DIR="${BASE_DIR}/../../tensorflow_libraries/models";
echo "MODELS_DIR => $MODELS_DIR";
PYTHONPATH="${MODELS_DIR}:${MODELS_DIR}/research/:${MODELS_DIR}/research/slim:$PYTHONPATH"

TRAINING_FOLDER="training"
INFERENCE_FOLDER="inference_graph"

if [ -z "$1" ]
  then
    echo "No argument supplied: Dataset folder name"
    exit 0
fi

if [ -z "$2" ]
  then
    echo "No argument supplied: model.ckpt number"
    exit 0
fi

python3 "${MODELS_DIR}"/research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path "$1"/pipeline_v2.config --trained_checkpoint_prefix "$1"/"${TRAINING_FOLDER}"/model.ckpt-"$2" --output_directory  "$1"/"${INFERENCE_FOLDER}"