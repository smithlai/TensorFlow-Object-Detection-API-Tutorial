#copy images and xmls into images folder (into subfolder train and test)

BASE_DIR="$(cd "$(dirname "$0")"; pwd)";
echo "BASE_DIR => $BASE_DIR";
MODELS_DIR="${BASE_DIR}/../models";
echo "MODELS_DIR => $MODELS_DIR";
export PYTHONPATH="${MODELS_DIR}:${MODELS_DIR}/research/:${MODELS_DIR}/research/slim:$PYTHONPATH"

if [ -z "$1" ]
  then
    echo "No argument supplied: Dataset folder name"
    exit 0
fi
#This will read all xmls into one csv file(train_labels and test_labels).
python3 xml_to_csv.py -s "$1"

#python3 pbtxt.py -s "$1"
#Combine the csv and images into one tfrecord file
python3 generate_tfrecord.py --csv_input="$1"/train_labels.csv --image_dir="$1"/train --output_path="$1"/train.record
python3 generate_tfrecord.py --csv_input="$1"/test_labels.csv --image_dir="$1"/test --output_path="$1"/test.record

#train data according to config file data/pipeline_v2.config
python3 ${MODELS_DIR}/research/object_detection/legacy/train.py --logtostderr --train_dir="$1"/training/ --pipeline_config_path="$1"/pipeline_v2.config


