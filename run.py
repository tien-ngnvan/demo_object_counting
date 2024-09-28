import json
import os
import cv2
import argparse
from glob import glob
import json

from processor import Processor
from logger import setup_logger, LoggerFormat


IMG_TYPES = [".jpg", ".jpeg", ".png"]
INFOR_RUN_FILE = "infor_run.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Human Tracking")
    parser.add_argument(
        "--input_folder_or_imgname",
        # default="sample",
        type=str,
        required=True,
        help="Path to the input image/video file. Support image (jpg, png, jpeg)",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="output",
        help="Path to the output video file",
    )
    parser.add_argument(
        "--yolodetect_path",
        type=str,
        default="weights/yolox_.onnx",
        help="Path to the human detection model",
    )
    parser.add_argument(
        "--yolo_classifier_path",
        type=str,
        default="weights/reid/REID_ghostnetv1.onnx",
        help="Path to the human reid model",
    )
    parser.add_argument(
        "--system_log_path",
        type=str,
        default="system_logs.txt",
        help="Path to the system log file",
    )
   
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    # Initialize processor and logger
    sys_logger = setup_logger(args.system_log_path, format=LoggerFormat.SYSTEM)
    processor = Processor(**vars(args))
    
    sys_logger.info(f"Processing image: {args.input_folder_or_imgname}")
    image = cv2.imread(args.input_folder_or_imgname)
    output = processor(image)
        