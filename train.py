from IPython import display
import ultralytics
from ultralytics import YOLO
from IPython.display import display, Image
from glob import glob
import torch

#torch.cuda.set_device(0)

''' [train dataset 다운로드]'''

# from roboflow import Roboflow
# rf = Roboflow(api_key="******")
# project = rf.workspace("yeeun-ayqkl").project("cooksaveproject")
# version = project.version(6)
# dataset = version.download("yolov8")


# from roboflow import Roboflow
# rf = Roboflow(api_key="U1jVQ7l0AIvIU7wnuodW")
# project = rf.workspace("testveg").project("test-veg")
# version = project.version(1)
# dataset = version.download("yolov8")


img_list = glob('/test-veg-1/train/images/*.jpg') # 트레인 이미지 경로
val_img_list = glob('/test-veg-1/test/images*.jpg') # 테스트 이미지 경로

with open('./test-veg-1/train.txt', 'w') as f:
    f.write('\n'.join(img_list) + '\n')

with open('./test-veg-1/test.txt', 'w') as f:
    f.write('\n'.join(val_img_list) + '\n')

#dataset.location='/home/ubuntu/CookSaveProject/test-veg-1'

model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
#model.to('cuda')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('*************************Using device:', device)
# print()
model.train(data="/home/ubuntu/CookSaveProject/test-veg-1/data.yaml", epochs=150, imgsz=416)





#data.yaml 파일에서 변경하기 

