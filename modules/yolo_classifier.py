import os
import sys
import cv2
import numpy as np
import torch.nn.functional as F
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))



class YoloClassifier:
    def __init__(self, model_path):
        self.load_model(model_path, model_type=model_path.split(".")[-1])

    def load_model(self, model_path, model_type):
        if model_type == "onnx":
            import onnxruntime as ort

            self.model = ort.InferenceSession(
                model_path,
                providers=[
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
            )
        elif model_type == "pt":
            import torch

            self.model = torch.load(model_path)
        else:
            raise ValueError("Model type not supported")
        
        try:
            import json
            
            meta = self.model.get_modelmeta().custom_metadata_map
            name = meta['names'][1:-1].split(',')
            name = [str(x.split(':')[-1].replace("'", "\n").strip()) for x in name]
   
            self.names = name
            
        except:
            self.names = self.model.names
        
        
    def preprocess(self, img, input_size, swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_img = (
                np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
            )
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def detect(
        self, img, test_size=(128, 128), conf_det=0.3, nmsthre=0.45, num_classes=2
    ):
        tensor_img, _ = self.preprocess(img, test_size)
        tensor_img = np.expand_dims(tensor_img, 0)
        
        outputs = self.model.run(None, {self.model.get_inputs()[0].name: tensor_img})[0]
        pred = F.softmax(torch.Tensor(outputs), dim=1)
        
        for i, prob in enumerate(pred):
            result = dict()
            top5i = prob.argsort(0, descending=True)[:5].tolist()
            for j in top5i:
                result.update({self.names[j]:prob[j].item()})
        
        cls_name = self.names[torch.argmax(prob)]
        
        return cls_name, result

if __name__ == '__main__':
    import glob
    
    model_path = os.path.join('weights', 'classification.onnx')
    classifier = YoloClassifier(model_path)
    
    img = cv2.imread(os.path.join('predict', 'opt5.jpg'))[:,:,::-1]
    classifier.detect(img)
    
    # for x in glob.glob('predict/*.jpg'):
    #     print("Detect sample: ", x.split('/')[-1])
    #     img = cv2.imread(x)
    #     classifier.detect(img)
    #     break
    