from ultralytics import YOLO
from urllib.request import  urlopen


if __name__=='__main__':
    # model = YOLO('/home/omnisky/hzp/Airs/yaml/ADDyolov8.yaml')
    model = YOLO('/home/omnisky/hzp/Airs/weights/Concat2L_conf0.0001.pt')
    print(model)

    # model.train(data='/home/omnisky/hzp/Airs/data/airs.yaml',batch=64, device=[0,1,2,3], epochs=300,lr0=0.01)

