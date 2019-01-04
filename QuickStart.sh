'''
#1. copy images and xmls into images folder (into subfolder "train" and "test" of images)
#2. download models
#3. modify pipeline.config
    -- specify model,pbtxt,tfrecords,numbers here
'''
BASE_DIR="$(cd "$(dirname "$0")"; pwd)";
echo "BASE_DIR => $BASE_DIR";
MODELS_DIR="${BASE_DIR}/../models";
echo "MODELS_DIR => $MODELS_DIR";

export PYTHONPATH="${MODELS_DIR}:${MODELS_DIR}/research/:${MODELS_DIR}/research/slim:${MODELS_DIR}/research/object_detection:$PYTHONPATH"

#This will read all xmls into one csv file(train_labels and test_labels).
python3 xml_to_csv.py

#Combine the csv and images into one tfrecord file
python3 generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
python3 generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record

#train data according to config file data/pipeline_v2.config
python3 ${MODELS_DIR}/research/object_detection/legacy/train.py --logtostderr --train_dir=training/ --pipeline_config_path=data/pipeline_v2.config


