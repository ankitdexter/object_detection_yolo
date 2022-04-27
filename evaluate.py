from imageai.Detection.Custom import DetectionModelTrainer

data_directory="data/trainval/"
model_path="data/trainval/models/detection_model-ex-012--loss-0059.259.h5"
json_path="data/trainval/json/detection_config.json"

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory=data_directory)
metrics = trainer.evaluateModel(model_path=model_path, json_path=json_path, iou_threshold=0.5, object_threshold=0.6, nms_threshold=0.5)