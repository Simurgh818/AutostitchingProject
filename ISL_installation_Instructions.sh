# This installation instruction is for In-Silico-Labeling paper (Christiansen et al 2018, https://github.com/google/in-silico-labeling)

# This instruction has been tested on CentOS 7, 16 GB RAM. 

# ==================================================================================
# Step1: check python and pip versions, we want python 3.6 installed
python3.6 -V 
pip3.6 -V 
# If version 3.6 is not installed install using the following instructions:
  #Step 1.1: Updating yum
  sudo yum update
  sudo yum install yum-utils
  sudo yum group mark install development

  #Step 1.2: python 3.6 installation
  sudo yum install https://centos7.iuscommunity.org/ius-release.rpm
  sudo yum install python36u
  #Test and verify successful installation: 
  python3.6 -V 
  #Step 1.2.1: install pip and devel
  sudo yum install python36u-pip
  sudo yum install python36u-devel

# ==================================================================================
#Step 2: create python3.6 virtual environment:
python3.6 -m venv SinaTesting
# instead of SinaTesting put specify the name and location for the virtual environment.
# To activate the virutal environment go to:
source SinaTesting/bin/activate
# To update the pip of the virtual environment from 9.0.1 to higher use:
(SinaTesting)$ pip install -U pip
# Step2.1: Install numpy wheel, six, protobuf, opencv-python libraries using pip:
(SinaTesting)$ pip install numpy wheel six protobuf opencv-python

# ==================================================================================
#Step 3: Install bazel: https://docs.bazel.build/versions/master/install-redhat.html
#Step 3.1: Download CentOS7 repo from FEDORA COPR (https://copr.fedorainfracloud.org/coprs/vbatts/bazel/) and copy it into /etc/yum.repos.d/
sudo yum install bazel

# ==================================================================================
#Step 4: Tensorflow installation:
#Step 4.1: standard installation
(SinaTesting)$ pip install -U tensorflow

#Step 4.2: Optional - installation from source could install optimized tensorflow for your machine's CPU
# Clone the git rpository in the directory you want to place it.
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
#Step 4.2.1: need to configure the installation first:
./configure
# specify the location of python3.6
# specify the location of the packages for it.
# jmalloc? n
# Google Cloud? n
# Amazon AWS? n
# Hadoop file sys? n
# Apache Kafka? n
# XLA JIT? n
# VERBS support? n
# GDR support? n
# OpenCl support? n
# CUDA support? n
# MPI support? n
# fresh release of clang? n
# ./WORKSPACE for android? n
# To have the configuration install the version opeitmized for your machine's CPU type:
-march=native
#Step 4.2.2: need to then build the packages: 
bazel build --config=opt//tensorflow/tools/pip-package:build-pip-package
#Step 4.2.3: installation
pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl

#Step 4.3: verify successful installation:
# python
# >>import tensorflow as tf 
# >>hello = tf.constant('Hello, Tensorflow')
# >>sess = tf.Session()
# >>print (sess.run(hello))
# if no error then good to go. You might get a warning that installation is not optimal for your CPU. It is a common warning. If you installed using standard installation, try installing from source. 

# ==================================================================================
#Step 5: Download data and unzip into isl subfolder:
unzip ~/isl/checkpoints.zip
unzip ~/isl/data_sample.zip

# ==================================================================================
#Step 6: Running the pretrained model on a seen images

#Step 6.1:Optional - if your computer has lower than 32 GB RAM edit the launch.py file in the isl subfolder so the program run its modules in serial rather than parallel. The parallel processing cause it to use too much RAM, and the computer then kills it.
# Change the following parameters values to 1 in launch.py:
# infer_size 
# process_batch_size
# process_shuffle_batch_num_threads
# preprocess_batch_capacity

# Also, choose a smaller crop size than 1500. 375 is the smallest size. 

# In addition, note I have the output and error messages saved as two text files. Please edit them based on the date, crop size and the number of images in the folder you are running. After testing and verification of successfully seeing images from before the training (seen and unseen images) and also after the training (seen and unseen images), you can take out the last two lines for the txt files. 

#Step 6.2: Running the pre-trained model.

# BASE_DIRECTORY: the working directory of the model. 
# alsologtostderr: makes the program print progress information in terminal. Remember that the last two lines I have added sends these information to output and error txt files. So you will not see any output in your terimal unless you take out the last two lines.
# mode EVAL_EVAL = uses the pretrained model
# stitch_crop_size: defult is 1500. Smallest one is 375.
# infer_channel_whitelist: list of fluorescent channels we wish to infer 

export BASE_DIRECTORY=/home/sinadabiri/venvs/in-silico-labeling-master/isl

bazel run isl:launch -- \
  --alsologtostderr \
  --base_directory $BASE_DIRECTORY \
  --mode EVAL_EVAL \
  --metric INFER_FULL \
  --stitch_crop_size 375\
  --restore_directory $BASE_DIRECTORY/checkpoints \
  --read_pngs \
  --dataset_eval_directory $BASE_DIRECTORY/data_sample/condition_b_sample \
  --infer_channel_whitelist DAPI_CONFOCAL,MAP2_CONFOCAL,NFH_CONFOCAL \
  	> Testing_output_preTrainedModel_8-13-18_375crop_16images.txt \
  	2> testing_error_preTrainedModel_8-13-18_375crop_16images.txt


#Step 6.3: Testing the pre-trained model on a new dataset (unseen images).

export BASE_DIRECTORY=/home/sinadabiri/venvs/in-silico-labeling-master/isl

bazel run isl:launch -- \
  --alsologtostderr \
  --base_directory $BASE_DIRECTORY \
  --mode EVAL_EVAL \
  --metric INFER_FULL \
  --stitch_crop_size 375 \
  --restore_directory $BASE_DIRECTORY/checkpoints \
  --read_pngs \
  --dataset_eval_directory $BASE_DIRECTORY/data_sample/condition_e_sample_B3 \
  --infer_channel_whitelist DAPI_CONFOCAL,CELLMASK_CONFOCAL \
  --noinfer_simplify_error_panels \
    > testing_output_postTrainedModel_newDataSet_8-13-18_375crop_B2images.txt \
    2> testing_error_postTrainedModel_newDataSet_8-13-18_375crop_B2images.txt
 
# ==================================================================================
#Step 7: Training the model

export BASE_DIRECTORY=/home/sinadabiri/SinaTesting3/in-silico-labeling/isl

bazel run isl:launch -- \
  --alsologtostderr \
  --base_directory $BASE_DIRECTORY \
  --mode TRAIN \
  --metric LOSS \
  --master "" \
  --restore_directory $BASE_DIRECTORY/checkpoints \
  --read_pngs \
  --dataset_train_directory $BASE_DIRECTORY/data_sample/condition_e_sample_B2  \
    > testing_output_Training_8-8-18_375crop_16image.txt \
    2> testing_error_Training_8-8-18_375crop_16image.txt

#Step 7.1: check the loss graphs to see how our model is training 

tensorboard --logdir $BASE_DIRECTORY

# Open the link that above line produces to see the graphs.

# ==================================================================================
#Step 7.2: To generate predictions:

export BASE_DIRECTORY=/home/sinadabiri/SinaTesting3/in-silico-labeling/isl


bazel run isl:launch -- \
  --alsologtostderr \
  --base_directory $BASE_DIRECTORY \
  --mode EVAL_EVAL \
  --metric INFER_FULL \
  --stitch_crop_size 375 \
  --read_pngs \
  --dataset_eval_directory $BASE_DIRECTORY/data_sample/condition_e_sample_B3 \
  --infer_channel_whitelist DAPI_CONFOCAL,CELLMASK_CONFOCAL \
  --noinfer_simplify_error_panels \
    > testing_output_PostTraining_8-14-18_375crop_16image.txt \
    2> testing_error_PostTraining_8-14-18_375crop_16image.txt

# ==================================================================================
# THE END