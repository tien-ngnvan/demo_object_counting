import cv2
import os
import glob

from modules.yolo_classifier import YoloClassifier
from modules.yolo_detect import YoloDetect


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
        nimg, _, _ = self.yolodetect.detect(img, draw=True)
        
        subpath = os.path.join('predict', '*.jpg')
        sub_files = glob.glob(subpath)
        
        class_name = []
        for x in sub_files:
            subimg = cv2.imread(x)[:,:,::-1]
            cls_name, _ = self.yoloclassifier.detect(subimg)
            class_name.append(cls_name)
        
        
        romano, x_men, unlabel = 0, 0, 0
        for i in class_name:
            if i.lower() == 'romano':
                romano += 1
            elif i.lower() == 'x_men':
                x_men += 1
            else:
                unlabel += 1
        objects = {"romano":romano, "x_men":x_men, "unlabel":unlabel}
        
        for i, (class_name, count) in enumerate(objects.items()):
            text = f'{class_name}: {count} objects'
            cv2.putText(img, text, (50, 50 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        
        cv2.imwrite('predictall.jpg', nimg)
    
    
if __name__ == '__main__':
    processor = Processor(
        yolodetect_path=os.path.join('weights', 'detection.onnx'),
        yolo_classifier_path=os.path.join('weights', 'classification.onnx'),
    )
    img = cv2.imread(os.path.join('samples', 'test5.jpg'))
    processor(img)
    # best case: 3, 6