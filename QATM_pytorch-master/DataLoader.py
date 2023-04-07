import torch
import cv2
from pathlib import Path
from torchvision import models , transforms , utils
import re, pytesseract 
import numpy as np
import pandas as pd 

# |이 코드는 PyTorch의 Dataset 클래스를 상속하여 이미지 데이터셋을 처리하는 클래스입니다.
# |
# |좋은 점:
# |- `__init__` 함수에서는 입력으로 받은 디렉토리 경로에 있는 이미지 파일들의 경로를 리스트로 저장합니다.
# |- `__len__` 함수에서는 데이터셋의 크기를 반환합니다. 이 함수는 데이터 로더에서 데이터셋의 크기를 파악하는 데 사용됩니다.
# |- `__getitem__` 함수에서는 인덱스를 입력으로 받아 해당 인덱스에 해당하는 이미지 파일을 읽어와 전처리를 수행합니다. 이 함수는 데이터 로더에서 데이터를 불러오는 데 사용됩니다.
# |
# |나쁜 점:
# |- `extract_draw_num` 함수가 정의되어 있지 않아서 이 함수가 어떤 역할을 하는지 파악하기 어렵습니다.
# |- `extract_draw_num` 함수에서 반환하는 `draw_num` 변수가 어떤 의미를 가지는지 주석이나 설명이 없어서 이 변수가 어떤 용도로 사용되는지 파악하기 어렵습니다.
# |- `image.size()` 함수가 호출되는 부분에서 `image` 변수가 PyTorch의 Tensor 형태인지, OpenCV의 Mat 형태인지 파악하기 어렵습니다. 이 부분에서 오류가 발생할 가능성이 있습니다.
# |
# |개선할 점:
# |- `extract_draw_num` 함수의 역할과 반환값에 대한 주석을 추가하여 코드의 가독성을 높입니다.
# |- `image.size()` 함수 대신 `image.shape` 속성을 사용하여 이미지의 크기를 파악합니다. 이렇게 하면 `image` 변수가 어떤 형태인지 파악할 필요가 없어서 코드의 가독성이 높아집니다.
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, template_dir_path , transform = None):
        
        self.transform = None
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])

        self.template_path = list(template_dir_path.iterdir())

        
    def __len__(self):
        return len(self.template_path)
    
    def __getitem__(self, idx):
   
        template_path = str(self.template_path[idx])

        image_raw = cv2.imread(template_path)
        
        image , draw_num = self.extract_draw_num(image_raw , template_path)
        if self.transform:
            image = self.transform(image).unsqueeze(0)

        image = image.type(torch.float32)
        
        return {'image': image, 
                    'image_raw': image_raw, 
                    'image_name': template_path, 
                    'image_h': image.shape[-3],
                   'image_w': image.shape[-2],
                   'draw_num' : draw_num,
            }
        
        
    def extract_draw_num(self , image , path):
        gray_scale = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
        _ , img_bin = cv2.threshold(gray_scale , 150 , 225 , cv2.THRESH_BINARY)
        img_bin = ~img_bin
        
        line_min_width = 100
        kernel_h = np.ones((1 ,line_min_width) , np.uint8)
        kernel_v = np.ones((line_min_width , 1) , np.uint8) 


        img_bin_h = cv2.morphologyEx(img_bin ,  cv2.MORPH_OPEN , kernel_h)
        img_bin_v = cv2.morphologyEx(img_bin , cv2.MORPH_OPEN , kernel_v)

        img_bin_final = img_bin_h|img_bin_v
        final_kernel  = np.ones((3,3) , np.uint8)
        img_bin_final = cv2.dilate(img_bin_final , final_kernel , iterations = 1) 
        
        _ , labels  , stats , _ = cv2.connectedComponentsWithStats(
                                                                ~img_bin_final ,
                                                                connectivity = 8,
                                                                ltype = cv2.CV_32S
                                                                )
          
        # new_stats = sorted(stats , key = lambda x : (x[0]  , x[1]))
        # left_bound = new_stats[-2][0]
        stats = sorted(stats , key = lambda x : (x[0] + x[2]  , x[1] + x[3]))

        left_bound = stats[-3][0]

        img = cv2.imread(path)

        graph = img[: , left_bound:]
        text = pytesseract.image_to_data(graph) 

        
        #list_result = re.findall(r'\s[A-Z]-\d{1,3}', text)
        list_result = re.findall(r'\b([A]-\w{1,3})\b', text)
        split_data = text.split('\n')

        if list_result:
            for data in split_data:
                if list_result[0] in data:
                    index_data = data.split('\t')
                    x , y , w , h , index = str(image.shape[1] - int(index_data[-4])) , str(image.shape[0] - int(index_data[-5])) , index_data[-3] , index_data[-2] , re.sub('[,.]' , '' ,  index_data[-1])
        else:
            index = 'None'
                
        crop_img = image[: , :left_bound]
  
        
        return crop_img , index
        #return crop_img , (x , y , w , h ,index)





class TemplateDataset(torch.utils.data.Dataset):
    def __init__(self, template_dir_path,image,thresh_csv=None, transform=None):
        self.transform = transform
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])

        self.template_path = list(template_dir_path.iterdir())
        
        self.image = image
        self.thresh_df = None
        if transform:
            self.image = self.transform(self.image).unsqueeze(0)
      
        
        if thresh_csv:
            self.thresh_df = pd.read_csv(thresh_csv)
            
        
    def __len__(self):
        return len(self.template_path)
    
    def __getitem__(self, idx):
   
        template_path = str(self.template_path[idx])
        
        template = cv2.imread(template_path)
    
        if self.transform:
            template = self.transform(template)
            
        thresh = 0.7
        if self.thresh_df is not None:
            if self.thresh_df.path.isin([template_path]).sum() > 0:
                thresh = float(self.thresh_df[self.thresh_df.path==template_path].thresh)
        
        return {'image': self.image,  
                    'template' : template.unsqueeze(0),
                    'template_name' : template_path,
                    'image_h': template.size()[-2],
                   'image_w': template.size()[-1],
                   'thresh': thresh
                    }
        
        
        
        
        