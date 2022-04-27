from imageai.Detection.Custom import DetectionModelTrainer

#providing path to the dataset folder and pretrained yolo model
data_directory="data/trainval/"
train_from_pretrained_model="data/pretrained-yolov3.h5"

#using imageai method to train
trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory=data_directory) 
trainer.setTrainConfig(object_names_array=["person", "car"], batch_size=16, num_experiments=20, train_from_pretrained_model=train_from_pretrained_model)
trainer.trainModel()