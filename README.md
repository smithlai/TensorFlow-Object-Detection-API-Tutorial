

# How To Train Object Detection API

I Believed that you've already know what the [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) is.

This tutorial is to provide a fast training script.

1. Install [anaconda](https://www.anaconda.com/)
2. git clone https://github.com/tensorflow/models.git
3. Create conda virtual environment include tensorflow and python

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

4. conda install Cython contextlib2 pillow lxml jupyter matplotlib

goto *models/research* folder

5. Protobuf
```
# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.
```

6. modifly the *model_main.py* in models
```
/research/object_detection/model_main.py
@@ -25,6 +25,8 @@ import tensorflow as tf
 from object_detection import model_hparams
 from object_detection import model_lib
 
+tf.logging.set_verbosity(tf.logging.INFO)
+
 flags.DEFINE_string(
     'model_dir', None, 'Path to output model directory '
     'where event and checkpoint files will be written.')
@@ -59,7 +61,7 @@ FLAGS = flags.FLAGS
 def main(unused_argv):
   flags.mark_flag_as_required('model_dir')
   flags.mark_flag_as_required('pipeline_config_path')
-  config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir)
+  config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir, log_step_count_steps=1)
```

7. modify the *MODELS_DIR* in MajiangQuickStart.sh

8. Gathering and labeling pictures (we have example in majian_data folder)

9. `MajiangQuickStart.sh majian_data`
This will read images in majian_data/train and majian_data/test, and output training process to training

10. To exporting the inference graph
Refers to `MajiangFrozen.sh`

## To train your own data
1. Create your own folder, e.g: "pocker"
2. Gathering and labeling pictures
3. separate the picture into *pocker/train* and *pocker/test*
4. list the classes in *pocker/predefined_classes.txt*
5. 
## To change the model zoo
1. Get your faverate model from [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

2. Untar model, copy to this project, and copy the pipeline.config to template/pipeline.template
3. Modify the num_classes: <tag_number>