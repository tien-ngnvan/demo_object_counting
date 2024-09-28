import cv2
import os
import glob

from modules.yolo_classifier import YoloClassifier
from modules.yolo_detect import YoloDetect


COLOR = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]

class Processor:
    def __init__(
        self,
        yolodetect_path,
        yolo_classifier_path,
        *args,
        **kwargs,
    ) -> None:        
        
        self.yolodetect = YoloDetect(yolodetect_path)
        self.yoloclassifier = YoloClassifier(yolo_classifier_path)
        self.args = args
        self.kwargs = kwargs
        self.mapper = None

    def __call__(self, img): 
        nimg, _, _ , logs = self.yolodetect.detect(img, draw=True)
        
        subpath = os.path.join('predict', '*.jpg')
        sub_files = glob.glob(subpath)
        class_name = []

        
        for x in sub_files:
            name = x.split('/')[-1]
            subimg = cv2.imread(x)[:,:,::-1]
            cls_name, result = self.yoloclassifier.detect(subimg)
            class_name.append(cls_name)

            # draw
            x1, y1, x2, y2 = logs[name].astype(int)
            prob_name = list(result.keys())[0]
            if prob_name.lower() == 'x-men':
                _name = 'XM'
                _color = COLOR[0]
            elif prob_name.lower() == 'romano':
                _name = 'RO'
                _color = COLOR[1]
            else:
                _name = "UK"
                _color = COLOR[2]
            prob_score = result[prob_name]
            cv2.rectangle(nimg, (x1, y1), (x2, y2), color=_color, thickness=2)
            cv2.putText(nimg,f"{_name}:{prob_score:.2f}",(x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _color, 2)
        
        romano, x_men, unlabel = 0, 0, 0
        for i in class_name:
            if i.lower() == 'romano':
                romano += 1
            elif i.lower() == 'x_men':
                x_men += 1
            else:
                unlabel += 1
        objects = {"ROmano":romano, "XMen":x_men, "UNknow":unlabel}
        
        for i, (class_name, count) in enumerate(objects.items()):
            text = f'{class_name}: {count} objects'
            if class_name.lower() == 'xmen':
                _color = COLOR[0]
            elif class_name.lower() == 'romano':
                _color = COLOR[1]
            else:
                _color = COLOR[2]
            cv2.putText(img, text, (50, 50 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 2, _color, 3)
        
        cv2.imwrite('predictall.jpg', nimg)
    
    
if __name__ == '__main__':
    processor = Processor(
        yolodetect_path=os.path.join('weights', 'detection.onnx'),
        yolo_classifier_path=os.path.join('weights', 'classification.onnx'),
    )
    img = cv2.imread(os.path.join('samples', 'test5.jpg'))
    processor(img)
    # best case: 3, 6