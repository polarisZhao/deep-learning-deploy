## Tensorflow-TensorRT

This is a repo about optimizing deep learning model using TensorRT. We demonstrate optimizing LeNet-like model and YOLOv3 model, and get 3.7x and 1.5x faster for the former and the latter, respectively, compared to the original models. For the details and how to run the code, see the video below.

 [YouTube video series](https://www.youtube.com/watch?v=AIGOSz2tFP8&list=PLkRkKTC6HZMwdtzv3PYJanRtR6ilSCZ4f)                            [Github Address](https://github.com/ardianumam/Tensorflow-TensorRT)

#### List of ipynb

- **1_convert_TF_to_TRT.ipynb**

- **2_inference_using_TensorRT-model.ipynb**
- **3_visualize_the_original_and_optimized_models.ipynb**
- **4_optimizing_YOLOv3_using_TensorRT.ipynb**

#### Dataset

Download (subset of) MNIST dataset [here](https://drive.google.com/file/d/1GOeU5T5EinT98VJsDbV0REyxEdgDwvio/view?usp=sharing), extract and put in folder `dataset`.

#### YOLOv3 Frozen Model

Download [here](https://drive.google.com/file/d/1tH6RCYXfsvS_BC2Z_zEd7mu4uMYW4dsr/view?usp=sharing), extract and put in folder `model/YOLOv3` 

