from imageai.Detection.Custom import CustomObjectDetection
from tqdm import tqdm
import os

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("data/trainval/models/detection_model-ex-012--loss-0059.259.h5")
detector.setJsonPath("data/trainval/json/detection_config.json")
detector.loadModel()
detector.__colors = (255,0,0)

def inferDir(in_dir, out_dir):
    imgs = os.listdir(in_dir)
    for img in tqdm(imgs):
        detections = detector.detectObjectsFromImage(input_image=in_dir + img, output_image_path=out_dir + img, minimum_percentage_probability=75, nms_treshold=0.4)


if __name__== '__main__': 
    #if we want to run over directory
    #inferDir(test_dir, output_dir)

    #if we want to pass selected images
    imgs = ['data/trainval/test/images/image_000000046.jpg',\
        'data/trainval/test/images/image_000001904.jpg',\
        'data/trainval/test/images/image_000001980.jpg',\
        'data/trainval/test/images/image_000001604.jpg',\
        'data/trainval/test/images/image_000000590.jpg',\
        'data/trainval/test/images/image_000001960.jpg']

    for img in tqdm(imgs):
        detections = detector.detectObjectsFromImage(input_image=img, output_image_path=f"images/{img.split('/')[-1]}", minimum_percentage_probability=75, nms_treshold=0.4)