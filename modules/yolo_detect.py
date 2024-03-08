import numpy as np
import cv2
import torch
import glob
import re
import os
from pathlib import Path
from modules.utils import non_max_suppression

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
  
  
class YoloDetect:
    def __init__(self, model_path):
        self.load_model(model_path)
        
    def load_model(self, model_path):
        model_type = model_path.split(".")[-1]
        
        if model_type == "onnx":
            import onnxruntime as ort

            self.model = ort.InferenceSession(
                model_path,
                providers=[
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
            )
        elif model_type == 'pt':
            import torch

            self.model = torch.load(model_path)
        else:
            raise ValueError("Model type not supported")
        
    def preprocess(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, 0)
        im = np.ascontiguousarray(im, dtype=np.float32)
        im /= 255
        
        return im, r, (dw, dh)
    
    def post_process(self, pred, ratio, dwdh):
        """_summary_

        Args:
            pred (_type_): _description_
            conf_thre (float, optional): _description_. Defaults to 0.7.
            conf_kpts (float, optional): _description_. Defaults to 0.9.
            get_layer (str, optional): _description_. Defaults to 'face'.

        Returns:
            _type_: [bbox, score, class_name]
        """
        
        if isinstance(pred, list):
            pred = np.array(pred)
            
        padding = dwdh*2
        det_bboxes, det_scores, det_labels  = pred[:, :4], pred[:, 4], pred[:, 5]
        det_bboxes = (det_bboxes[:, 0::] - np.array(padding)) /ratio
        
        det_bboxes = det_bboxes[:,:].numpy().astype(int)

        return det_bboxes, det_scores, det_labels
    
    def detect(self, img, test_size=(640, 640), conf_det=0.6, nmsthre=0.45, get_layer=None, draw=False):
        tensor_img, ratio, dwdh = self.preprocess(img, test_size, auto=False)
        
        # inference head, face
        pred = self.model.run([], {self.model.get_inputs()[0].name: tensor_img})[0]
        pred = torch.Tensor(pred)
        outputs = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]
        bboxes, scores, _ = self.post_process(outputs, ratio, dwdh)
        
        if draw:
            x_offset, y_offset = 0.2, 0.15
            points = [
                self.draw_line(img, x_offset, y_offset, 1 - x_offset, y_offset),
                self.draw_line(img, 1 - x_offset, y_offset, 1 - x_offset, 1 - y_offset),
                self.draw_line(img, 1 - x_offset, 1 - y_offset, x_offset, 1 - y_offset),
                self.draw_line(img, x_offset, 1 - y_offset, x_offset, y_offset)
            ]

            points = list(dict.fromkeys(points))
            img = self.draw_bounding_boxes(img, bboxes, Polygon(points), saved_crop=True)
            
        
        return img, bboxes, scores


    def draw_bounding_boxes(self, image, boxes, polygon, saved_crop=True, opt_path='predict'):
        def is_inside(polygon, centroid):
            centroid = Point(centroid)
            return polygon.contains(centroid)
        
        w = image.shape[1]
        h = image.shape[0]
        
        ppath = Path(opt_path)
        if os.path.exists(ppath) and ppath.is_dir():
            import shutil
            
            shutil.rmtree(ppath)
            
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]

            if is_inside(polygon, (x1 / w, y1 / h)) and is_inside(polygon, (x2 / w, y2 / h)):
                if saved_crop:
                    crop = image[y1:y2, x1:x2]
                    output_file = str(self.increment_path(f'{opt_path}/opt.jpg', mkdir=True).with_suffix('.jpg'))
                    cv2.imwrite(output_file, crop)
                    
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return image

    def increment_path(self, path, exist_ok=False, sep='', mkdir=False):
        path = Path(path)
        if path.exists() and not exist_ok:
            suffix = path.suffix
            path = path.with_suffix('')
            dirs = glob.glob(f"{path}{sep}*")
            matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
            i = [int(m.groups()[0]) for m in matches if m]
            n = max(i) + 1 if i else 2
            path = Path(f"{path}{sep}{n}{suffix}")
        dir = path if path.suffix == '' else path.parent
        if not dir.exists() and mkdir:
            dir.mkdir(parents=True, exist_ok=True)
        return path
    
    def draw_line(self, image, xf1, yf1, xf2, yf2):
        w = image.shape[1]
        h = image.shape[0]

        start_point = (int(w * xf1), int(h * yf1))
        end_point = (int(w * xf2), int(h * yf2))

        cv2.line(image, start_point, end_point, (255, 0, 0), 7)

        return xf1, yf1
        

        
if __name__ == '__main__':
    import os
    
    yolo_hf = YoloDetect(
        model_path=os.path.join('weights', 'detection.onnx')
    )
    
    img = cv2.imread(os.path.join("samples", "test.png"))
    
    bboxes, score = yolo_hf.detect(img, draw=True)

    