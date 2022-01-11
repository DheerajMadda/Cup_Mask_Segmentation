# Cup_Mask_Segmentation


```
docker pull dheerajmadda/cup_mask_segmentation
docker run -p 8080:8080 dheerajmadda/cup_mask_segmentation
```

###  Description

 The aim of this project is to mask the cup within the input image.
 
 
### To run

Make sure you have necessary libraries installed

A) Clone the repo 

B) Run the cells of the jupyter notebook

## Want to build an image? 
1) Clone the repo -> cd Dockerize
2) docker build -t <image_name> .

#### Run as a container
```
😊 docker run -p 8080:8080 <image_name>
```
#### Go to the localhost:8080, you will see the Streamlit application running! 🙌


### How to Setup Tensorflow Object Detection Framework?
```
conda create -n <your_env_name> python=3.6
conda activate <your_env_name>
clone https://github.com/tensorflow/models/tree/v1.13.0
pip install pillow lxml Cython contextlib2 jupyter matplotlib pandas opencv-python tensorflow==1.14.0
conda install -c anaconda protobuf
protoc models/research/object_detection/protos/*.proto --python_out=.
python models/research/setup.py install
``` 
