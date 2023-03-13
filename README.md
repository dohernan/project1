# Object Detection in an Urban Environment

## Data

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).

[OPTIONAL] - The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. We have already provided the data required to finish this project in the workspace, so you don't need to download it separately.

## Structure

### Data

The data you will use for training, validation and testing is organized as follow:
```
/home/workspace/data/waymo
	- training_and_validation - contains 97 files to train and validate your models
    - train: contain the train data (empty to start)
    - val: contain the val data (empty to start)
    - test - contains 3 files to test your model and create inference videos
```
The `training_and_validation` folder contains file that have been downsampled: we have selected one every 10 frames from 10 fps videos. The `testing` folder contains frames from the 10 fps video without downsampling.
```
You will split this `training_and_validation` data into `train`, and `val` sets by completing and executing the `create_splits.py` file.


### Experiments
The experiments folder will be organized as follow:
```
experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file
    - experiment0/ - create a new folder for each experiment you run
    - experiment1/ - create a new folder for each experiment you run
    - experiment2/ - create a new folder for each experiment you run
    - label_map.pbtxt
    ...
```

## Prerequisites

### Local Setup

For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the [build directory](./build).

Follow [the README therein](./build/README.md) to create a docker container and install all prerequisites.

### Download and process the data

**Note:** ”If you are using the classroom workspace, we have already completed the steps in the section for you. You can find the downloaded and processed files within the `/home/workspace/data/preprocessed_data/` directory. Check this out then proceed to the **Exploratory Data Analysis** part.

The first goal of this project is to download the data from the Waymo's Google Cloud bucket to your local machine. For this project, we only need a subset of the data provided (for example, we do not need to use the Lidar data). Therefore, we are going to download and trim immediately each file. In `download_process.py`, you can view the `create_tf_example` function, which will perform this processing. This function takes the components of a Waymo Tf record and saves them in the Tf Object Detection api format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). We are already providing the `label_map.pbtxt` file.

You can run the script using the following command:
```
python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}
```

You are downloading 100 files (unless you changed the `size` parameter) so be patient! Once the script is done, you can look inside your `data_dir` folder to see if the files have been downloaded and processed correctly.

### Classroom Workspace

In the classroom workspace, every library and package should already be installed in your environment. You will NOT need to make use of `gcloud` to download the images.

## Instructions

### Exploratory Data Analysis

You should use the data already present in `/home/workspace/data/waymo` directory to explore the dataset! This is the most important task of any machine learning project. To do so, open the `Exploratory Data Analysis` notebook. In this notebook, your first task will be to implement a `display_instances` function to display images and annotations using `matplotlib`. This should be very similar to the function you created during the course. Once you are done, feel free to spend more time exploring the data and report your findings. Report anything relevant about the dataset in the writeup.

Keep in mind that you should refer to this analysis to create the different spits (training, testing and validation).


### Create the training - validation splits
In the class, we talked about cross-validation and the importance of creating meaningful training and validation splits. For this project, you will have to create your own training and validation sets using the files located in `/home/workspace/data/waymo`. The `split` function in the `create_splits.py` file does the following:
* create three subfolders: `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`
* split the tf records files between these three folders by symbolically linking the files from `/home/workspace/data/waymo/` to `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`

Use the following command to run the script once your function is implemented:
```
python create_splits.py --data-dir /home/workspace/data
```

### Edit the config file

Now you are ready for training. As we explain during the course, the Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.

### Training

You will now launch your very first experiment with the Tensorflow object detection API. Move the `pipeline_new.config` to the `/home/workspace/experiments/reference` folder. Now launch the training process:
* a training process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
Once the training is finished, launch the evaluation process:
* an evaluation process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```

**Note**: Both processes will display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using
`CTRL+C`.

To monitor the training, you can launch a tensorboard instance by running `python -m tensorboard.main --logdir experiments/reference/`. You will report your findings in the writeup.

### Improve the performances

Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup.

Keep in mind that the following are also available:
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
* experiment with the architecture. The Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many architectures. Keep in mind that the `pipeline.config` file is unique for each architecture and you will have to edit it.

**Important:** If you are working on the workspace, your storage is limited. You may to delete the checkpoints files after each experiment. You should however keep the `tf.events` files located in the `train` and `eval` folder of your experiments. You can also keep the `saved_model` folder to create your videos.


### Creating an animation
#### Export the trained model
Modify the arguments of the following function to adjust it to your models:

```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/
```

This should create a new folder `experiments/reference/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path /data/waymo/testing/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif
```

## Submission Template

### Project overview
The objective of this project is to explore and augment a given driving image dataset and test a Single Shot Detector (SSD) Resnet50 Convolutional Neural Network (CNN) on the augmented dataset. The project involves changing different parameters to improve the performance of the CNN. The driving image dataset is a collection of images taken from a car's front-facing camera.

The first step in the project is to explore the characteristics of the images in the dataset. This involves analyzing the resolution, color distribution, and other properties of the images. Understanding the characteristics of the images helps in selecting appropriate augmentation techniques that can enhance the dataset's diversity.

After analyzing the image characteristics, the next step is to augment the dataset using different techniques. Some of the techniques that can be used to augment the dataset include rotation, cropping, flipping, and adjusting the brightness and contrast. By augmenting the dataset, we can improve the CNN's ability to generalize to new images.

Once the dataset is augmented, the next step is to test the SSD Resnet50 CNN on the augmented dataset. The CNN is trained on the augmented dataset and tested on a validation set. The performance of the CNN is measured in terms of accuracy, precision, recall, and F1-score.

Finally, the project involves changing different parameters in the CNN to improve its performance. The parameters that can be changed include the learning rate, batch size, number of epochs, and the optimizer used. By changing these parameters, we can improve the CNN's ability to detect objects in the images.

### Set up

The config files are placed in experiments/reference
old is the initial one with some augmentations
pipeline2.config uses the rms_prop_optimizer 
pipeline3.config uses adam_optimizer 


### Dataset
#### Dataset analysis
In this task, I programmed a function in Python to visualize the images and bounding boxes of the given dataset. After that, you developed functions to analyze the darkness of the images, their sizes, the variance to locate the low visibility images, and the density of objects in the image.

Unfortunately, due to the poor workspace provided, I was unable to get any graphic representation of the data.

Analyzing the darkness of images is important because it can affect the accuracy of object detection models.

Analyzing the sizes and density of objects in the images can help us understand how objects are distributed in the images. This can help us optimize the object detection model to better identify and locate objects in the images.

### Training
#### Reference experiment
I trained the reference CNN and saw the "not so good" results. Unfortunately I didnt make any captures of that because I didnt know I would need them later for the submission, and later when I tried to train it again to make the captures and documentation, the workspace didnt work anymore because of a lot of memory problems, performance problems... and it was impossible for me to do it.

#### Improve on the reference
1. Data Augmentations

Data augmentation techniques are an essential component of training deep learning models, particularly in the field of object recognition for autonomous driving. These techniques are used to create additional training data from the existing dataset by applying various transformations to the original images. This has the advantage of improving the generalization capability of the model, making it more robust and less prone to overfitting.

Horizontal_flip is used to create a mirror image of the original image, which is particularly useful for object detection in autonomous driving scenarios where objects can appear from any direction.

Adjusting brightness and contrast helps to simulate different lighting conditions, which is crucial in the real-world scenarios as lighting conditions can vary widely.

Gaussian noise is used to simulate the effect of sensor noise, which is present in real-world datasets. By adding this type of noise to the images, the model becomes more robust to noise in the input data.

Adjusting saturation is important for dealing with images that have color variations due to different lighting conditions.

Image scaling and cropping techniques can be used to simulate different perspectives of the same object, which can help the model to better understand the object's features and characteristics. This is particularly important in autonomous driving scenarios where the same object can appear in different sizes and orientations.

Overall, using these data augmentation techniques can improve the accuracy and robustness of the object recognition model for autonomous driving.

Using the data augmentations the performance of the CNN improved significantly
![image](https://user-images.githubusercontent.com/38068231/224650704-22dc0e51-f0d9-450d-928f-b2bbb4adf7e5.png)

2. Improving the CNN
In this task where I tried to improve an SSD Resnet50 model, I experimented with different optimizers, namely SGD and Adam, to see their impact on the model's performance.

First, I tried the SGD optimizer and obtained very good results. This means that the model's performance improved significantly, possibly because the optimizer was better suited for the problem at hand. SGD works well with large datasets and can converge faster than other optimizers.
![image](https://user-images.githubusercontent.com/38068231/224650790-4826c7d7-23fa-4a18-8c92-2dc3bcc1acb7.png)

Next, I tried the Adam optimizer but got not so good results. This suggests that the optimizer may not have been the best choice for this problem. Adam optimizer is known for its ability to converge quickly, especially when the gradients are sparse, but it may not always lead to optimal results in all cases.

![image](https://user-images.githubusercontent.com/38068231/224651431-1ff3b095-8c71-411d-ac85-b6695e59ead6.png)


Overall, the experiment highlights the importance of trying out different optimization techniques to achieve the best results for a given problem. However, it's important to keep in mind that the performance of the model depends on various factors, including the dataset, hyperparameters, and the optimization algorithm, and there may not be a one-size-fits-all solution.
