# demo_object_counting

## Installed
```pip install -r requirements.txt```

## Download checkpoint
You need to get detection.onnx and classifier.onnx model weights [here](https://drive.google.com/drive/folders/1Xm4nr-Jeyl3WeMpv8ra2kfcSMlU4fVw9?usp=sharing) and then put ckpt on weights folder.
```
|__weights
|  |__ detection.onnx
|  |__ classification.onnx
|__run.py
```

### Demo
```
python run.py --input_folder_or_imgname samples/test3.jpg --yolodetect_path weights/detection.onnx --yolo_classifier_path weights/classification.onnx  
```