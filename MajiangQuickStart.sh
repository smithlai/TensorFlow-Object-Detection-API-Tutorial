#copy images and xmls into images folder (into subfolder train and test)

#BASE_DIR="$(cd "$(dirname "$0")"; pwd)";
#echo "BASE_DIR => $BASE_DIR";
MODELS_DIR="/home/ml/workspace/smith/MachineLearning/models";
echo "MODELS_DIR => $MODELS_DIR";
PYTHONPATH="${MODELS_DIR}:${MODELS_DIR}/research/:${MODELS_DIR}/research/slim:$PYTHONPATH"

TRAINING_FOLDER="training"


if [ -z "$1" ]
  then
    echo "No argument supplied: Dataset folder name"
    exit 0
fi
#This will read all xmls into one csv file(train_labels and test_labels).
python xml_to_csv.py -s "$1"

python pbtxt.py -s "$1"
#Combine the csv and images into one tfrecord file
python "$1"/generate_tfrecord.py --csv_input="$1"/train_labels.csv --image_dir="$1"/train --output_path="$1"/train.record
python "$1"/generate_tfrecord.py --csv_input="$1"/test_labels.csv --image_dir="$1"/test --output_path="$1"/test.record

killall tensorboard
tensorboard  --logdir "$1"/"${TRAINING_FOLDER}"/ &
#train data according to config file data/pipeline_v2.config
#python ${MODELS_DIR}/research/object_detection/legacy/train.py --logtostderr --train_dir="$1"/"${TRAINING_FOLDER}"/ --pipeline_config_path="$1"/pipeline_v2.config

python ${MODELS_DIR}/research/object_detection/model_main.py --pipeline_config_path="$1"/pipeline.config --model_dir="$1"/"${TRAINING_FOLDER}"/ --num_train_steps=50000 --sample_1_of_n_eval_examples=100 --alsologtostderr

