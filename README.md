# Object Detection Model Training on car-person Dataset

## 1 Introduction to the task in hand
The task is to build an Object Detection model which learns the detection from the given data. There are two classes: **person** and **car**. We are provided with images as well as its annotations. 
* There are 2239 images
* Annotation format is COCO


## 2 Solution
We will Fine tune YOLO-V3 model using open-source repository named **imageai**. This repository wrote YOLO network in Tensorlfow so there will be no need to install any other packages except Tensorflow. This repository requires the annotation to be in PASCAL-VOC format and for each image there should be seperate annotation file written in **PASCAL-VOC XML**. After getting the format of annotation ready we will split the dataset into train test and validation with 60% 20% and 20% as split shares. After this is done we will download the **pretrained-yolov3.h5** file and use this as baseline for fine tuning the model to our usecase. After this is done we will move on to testing/analysing/visualizing the performance of the model.

## 2.0 Python packages required
Follow the `requirements.txt` file for reference or simply `pip install -r requirements.txt`
## 2.1 Assumption
Before proceeding to the solution here are some assumptions that we take:
* We assume that the annotation is done accurately and there is very less or no human error done during the annotation process. 
* We assume that all the images are in good shape and no images are corrupted in terms of jpg or png encoding. 
* Another assumption but less concerning is that images have three channels viz. RGB

## 2.2 Approach
* We will Fine tune YOLO-V3 model using open-source repository named **imageai**
* **imageai** requires the annotation to be in PASCAL-VOC format and for each image there should be seperate annotation file written in **PASCAL-VOC XML**
* Split the dataset into train test and validation
* Download pretrained YOLO model and fine tune on train and validation dataset
* Check the performance of the model 
* Visualize the results

## 2.2.1 About imageai repository 
Quoting the imagai documentation: "ImageAI is a python library built to empower developers, reseachers and students to build applications and systems with self-contained Deep Learning and Computer Vision capabilities using simple and few lines of code. This documentation is provided to provide detailed insight into all the classes and functions available in ImageAI, coupled with a number of code examples."
* ImageAI provides 4 different algorithms and model types to perform image prediction, trained on the ImageNet-1000 dataset. The 4 algorithms provided for image prediction include MobileNetV2, ResNet50, InceptionV3 and DenseNet121.
* The arrangement of dataset folder structure should look like this example:
    * <div class="highlight"><pre><span></span><span class="o">&gt;&gt;</span> <span class="n">train</span>    <span class="o">&gt;&gt;</span> <span class="n">images</span>       <span class="o">&gt;&gt;</span> <span class="n">img_1</span><span class="o">.</span><span class="n">jpg</span>  <span class="p">(</span><span class="n">shows</span> <span class="n">Object_1</span><span class="p">)</span>
                  <span class="o">&gt;&gt;</span> <span class="n">images</span>       <span class="o">&gt;&gt;</span> <span class="n">img_2</span><span class="o">.</span><span class="n">jpg</span>  <span class="p">(</span><span class="n">shows</span> <span class="n">Object_2</span><span class="p">)</span>
                  <span class="o">&gt;&gt;</span> <span class="n">images</span>       <span class="o">&gt;&gt;</span> <span class="n">img_3</span><span class="o">.</span><span class="n">jpg</span>  <span class="p">(</span><span class="n">shows</span> <span class="n">Object_1</span><span class="p">,</span> <span class="n">Object_3</span> <span class="ow">and</span> <span class="n">Object_n</span><span class="p">)</span>
                  <span class="o">&gt;&gt;</span> <span class="n">annotations</span>  <span class="o">&gt;&gt;</span> <span class="n">img_1</span><span class="o">.</span><span class="n">xml</span>  <span class="p">(</span><span class="n">describes</span> <span class="n">Object_1</span><span class="p">)</span>
                  <span class="o">&gt;&gt;</span> <span class="n">annotations</span>  <span class="o">&gt;&gt;</span> <span class="n">img_2</span><span class="o">.</span><span class="n">xml</span>  <span class="p">(</span><span class="n">describes</span> <span class="n">Object_2</span><span class="p">)</span>
                  <span class="o">&gt;&gt;</span> <span class="n">annotations</span>  <span class="o">&gt;&gt;</span> <span class="n">img_3</span><span class="o">.</span><span class="n">xml</span>  <span class="p">(</span><span class="n">describes</span> <span class="n">Object_1</span><span class="p">,</span> <span class="n">Object_3</span> <span class="ow">and</span> <span class="n">Object_n</span><span class="p">)
      </span><span class="o">&gt;&gt;</span> <span class="n">validation</span>   <span class="o">&gt;&gt;</span> <span class="n">images</span>       <span class="o">&gt;&gt;</span> <span class="n">img_151</span><span class="o">.</span><span class="n">jpg</span> <span class="p">(</span><span class="n">shows</span> <span class="n">Object_1</span><span class="p">,</span> <span class="n">Object_3</span> <span class="ow">and</span> <span class="n">Object_n</span><span class="p">)</span>
                      <span class="o">&gt;&gt;</span> <span class="n">images</span>       <span class="o">&gt;&gt;</span> <span class="n">img_152</span><span class="o">.</span><span class="n">jpg</span> <span class="p">(</span><span class="n">shows</span> <span class="n">Object_2</span><span class="p">)</span>
                      <span class="o">&gt;&gt;</span> <span class="n">images</span>       <span class="o">&gt;&gt;</span> <span class="n">img_153</span><span class="o">.</span><span class="n">jpg</span> <span class="p">(</span><span class="n">shows</span> <span class="n">Object_1</span><span class="p">)</span>
                      <span class="o">&gt;&gt;</span> <span class="n">annotations</span>  <span class="o">&gt;&gt;</span> <span class="n">img_151</span><span class="o">.</span><span class="n">xml</span> <span class="p">(</span><span class="n">describes</span> <span class="n">Object_1</span><span class="p">,</span> <span class="n">Object_3</span> <span class="ow">and</span> <span class="n">Object_n</span><span class="p">)</span>
                      <span class="o">&gt;&gt;</span> <span class="n">annotations</span>  <span class="o">&gt;&gt;</span> <span class="n">img_152</span><span class="o">.</span><span class="n">xml</span> <span class="p">(</span><span class="n">describes</span> <span class="n">Object_2</span><span class="p">)</span>
                      <span class="o">&gt;&gt;</span> <span class="n">annotations</span>  <span class="o">&gt;&gt;</span> <span class="n">img_153</span><span class="o">.</span><span class="n">xml</span> <span class="p">(</span><span class="n">describes</span> <span class="n">Object_1</span><span class="p">)</span>
</pre></div>

    
## 2.2.2 Converting Annotation file to Desired Format
* We have to convert the COCO Format annotation to PASCAL-VOC XML format for which we used the notebook convert_annot_to_pascal_voc.ipynb or coco_to_pascal.py
    * COCO Format:
        * ![coco_format.png](https://github.com/ankitdexter/object_detection_yolo/blob/main/images/coco_format.png)
    * Pascal-VOC Format:
        * ![pascal_format.png](https://github.com/ankitdexter/object_detection_yolo/blob/main/images/pascal_format.png)
    * Code: Refer notebook `convert_annot_to_pascal_voc.ipynb` or `coco_to_pascal.py`

## 2.2.3 Train-Test-Validation Split
We are randomly splitting the dataset into 60% 20% 20% for train validation and test respectively
* Train images: 1345, Test images: 447, Validation images: 447
* Code: refer `splitting_dataset.ipynb` or `split_dataset.py`

## 2.2.4 Performance Metric
We are usin mAP as key performance metric which will penalize on model. 
* image here 
## 2.2.5 Training 
We are training our dataset with batch size 16 and epochs 30 using GPU accelerator Nvidia-TeslaP40
* Code: `train.py`
* Output: `trianOutput.py`
* loss vs epoch curve (blue: validation, red: train)
   * ![epoch_loss.svg](https://github.com/ankitdexter/object_detection_yolo/blob/main/images/epoch_loss.svg)
* best epoch to choose will be 12 as there is small gap between train and validation loss 
* Download model: https://drive.google.com/file/d/1F1WreKNRHfuGKKOBpZAuGOMkovL_zuvz/view?usp=sharing
* Download json_config: https://drive.google.com/file/d/1IMEslOfszyyePKqy36odXWYnL2ILwqWj/view?usp=sharing

## 2.2.6 Evaluation of trained model on test dataset(results)
We are choosing our final model on 12th epoch which has loss of 59.29
* Code: `evaluate.py`
* Output: `evaluateOutput.py`
* Evaluation samples = 447
* Using IoU = 0.5
* Using Object Threshold = 0.6
* Using Non-Maximum Suppression = 0.5
* Average Precision on car = 0.3631
* Average Precision on person = 0.2785
* Final mAP = 0.3208

## 2.2.7 Testing on Images (Visualization)
* Code: `inference.py`
* Results: 
   * ![image_000000046.jpg](https://github.com/ankitdexter/object_detection_yolo/blob/main/images/image_000000046.jpg)
   * ![image_000000590.jpg](https://github.com/ankitdexter/object_detection_yolo/blob/main/images/image_000000590.jpg)
   * ![image_000001604.jpg](https://github.com/ankitdexter/object_detection_yolo/blob/main/images/image_000001604.jpg)
   * ![image_000001904.jpg](https://github.com/ankitdexter/object_detection_yolo/blob/main/images/image_000001904.jpg)
   * ![image_000001960.jpg](https://github.com/ankitdexter/object_detection_yolo/blob/main/images/image_000001960.jpg)
   * ![image_000001980.jpg](https://github.com/ankitdexter/object_detection_yolo/blob/main/images/image_000001980.jpg)

## 2.2.8 Recommendations
* We can use newer version of YOLO model
* We can go for augmentations to increase the dataset size by using Albumentations library (https://albumentations.ai/)
* Try runninng for more epochs and look for model to converge further

## 2.2.9 Useful links
* https://github.com/OlafenwaMoses/ImageAI - Model Training Helper Library
* https://albumentations.ai/ - We can use this repo to augment our dataset
