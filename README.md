# faster_rcnn_logo

Application of faster_rcnn in logo detection case

Setup by Romuald FOTSO

## Introduction:

This project aims to use the py_faster_rcnn tool to detect logo in real scene image. If your are interessed by the original project (python), please feel free to have a look [here](https://github.com/rbgirshick/py-faster-rcnn). We have setup a small dataset with only 20 classes, and launch a training with the VGG16 architecture. This repository involves all required files to launch a train or test process by your own. Indeed this page does not present the best model performance on our dataset, it just shows one more use case of the py_faster_rcnn project.

![example](data/readme_img/example1.jpg)

## Datasets:

We have setup our own dataset (20 cls) based on pascal_voc schema.
Required files are available here:
  * [ROMYNY Logo 2016 : images+sets+annotations](https://drive.google.com/open?id=0B_Rjj_NgCayPS3NPamlydGZzcE0)
  * [ROMYNY Logo 2016 : labels](https://drive.google.com/open?id=0B_Rjj_NgCayPUXplX0tEazdmNnM)
  
## Hardwares/Softwares:
    OS: Ubuntu 16.04 64 bit
    GPU: Nvidia GTX 950M 4G
    Cuda 8.0
    CuDNN 3.0.8
    Python 2.7.12
    OpenCV 3.1.0

## Prerequisites:

  1. Caffe [prerequisites](http://caffe.berkeleyvision.org/installation.html#prequequisites)
  2. Python's packages (requirements.txt)
  
## Installation:

To install this project, please follow the steps below:

1. [Install OpenCV](http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/)

2. Download the repository:

    ```
    $ cd /opt
    $ sudo git clone --recursive https://github.com/romyny/faster_rcnn_logo.git
    $ cd faster_rcnn_logo
    ```
    
3. Install all externals repositories required:

    ```
    $ cd caffe-faster-rcnn
    ```
    
    Adjust Makefile.config, then
    
    ```
    $ sudo mkdir build
    $ cd build
    $ cmake ..
    $ make -j4
    $ make install
    $ cd ../..
    ```
    
    Install project's libs
    
    ```
    $ cd libs
    $ make
    $ cd ..
    ```
    
 4. Install python's packages required:

    ```
    for req in $(cat caffe-faster-rcnn/requirements.txt); do pip install $req; done
    ```
    
Get the data and models required:
1. Download the data and uncompress in 'data'
  * logo.20cls: [Google Drive](https://drive.google.com/open?id=0B_Rjj_NgCayPS3NPamlydGZzcE0) -> data/VOCdevkit2007

2. Download the models and uncompress in 'data/logo_models'
  * logo_models (VGG16): [Google Drive](https://drive.google.com/open?id=0B_Rjj_NgCayPRV9uWDhMM0lpcEk) -> data/logo_models
  * logo_models (RestNet-101): coming soon
  
## Experiments:

1. Run the demo: 
    
    ```
    $ cd $faster_rcnn_logo_DIR
    $ python tools/demo.py 
    ```
    Note: this script will be performed on the test set
    
    For help use the command: python tools/demo.py --help
    
2. Launch train & test process:

    ```
    $ cd $faster_rcnn_logo_DIR/experiments/scripts
    $ bash faster_rcnn_end2end_logo.sh 0 VGG16 pascal_voc \
      --set EXP_DIR logo_20cls RNG_SEED 42 TRAIN.SCALES "[400,500,600,700]"
    ```
    
## Our results
The model generated at the 30th iteration gives a mAP^0.5 = 0.6408

All results related to demo dataset will be saved at 'data/demo_out'

![example](data/readme_img/example2.jpg)

## Contact

Please feel free to leave suggestions or comments to Romuald FOTSO (romyny9096@gmail.com)    
