import torch
import cv2
from pathlib import Path
from torchvision import models , transforms , utils
import copy ,re
import numpy as np
import pandas as pd 
import easyocr
import natsort


def get_bound(stats):
    sorted_stats = sorted(stats , key = lambda x : (x[0] + x[2]  , x[1] + x[3]))
    left_bound = sorted_stats[-3][0]
    down_bound = sorted_stats[-3][1]
    right_bound = sorted_stats[-2][0]
    upper_bound = sorted_stats[-2][1]
    
    return (left_bound , down_bound , right_bound , upper_bound)


def get_stats(img):
        gray_scale = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
        _ , img_bin = cv2.threshold(gray_scale , 150 , 225 , cv2.THRESH_BINARY)
        img_bin = ~img_bin
        
        line_min_width = 100
        kernel_h = np.ones((1 ,line_min_width) , np.uint8)
        kernel_v = np.ones((line_min_width , 1) , np.uint8) 


        img_bin_h = cv2.morphologyEx(img_bin ,  cv2.MORPH_OPEN , kernel_h)
        img_bin_v = cv2.morphologyEx(img_bin , cv2.MORPH_OPEN , kernel_v)

        img_bin_final = img_bin_v | img_bin_h
        
        final_kernel  = np.ones((3,3) , np.uint8)
        img_bin_final = cv2.dilate(img_bin_final , final_kernel , iterations = 1)
        
        _ , labels  , stats , _ = cv2.connectedComponentsWithStats(
        ~img_bin_final ,
        connectivity = 8,
        ltype = cv2.CV_32S)

        new_stats = sorted(stats , key = lambda x : (x[0] + x[2]  , x[1] + x[3]))
        return new_stats
    
    

# 이미지 클래스는 아래와 같은 결과를 지니고 있다.
# ImageDataset:
#       image : symbol detect를 수행할 이미지
#       image_raw : 전처리를 하지 않은 원래 이미지
#       image_h , image_w : image의 높이와 너비
#       bounds : 표객체를 제외한 이미지의 경계선 , (upper_bound , down_bound , right_bound , left_bound) 순서이다.
#       draw_num : 해당 페이지 도면의 번호
#       page_num : 해당 페이지의 번호 

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, Image_dir_path , transform = None):
        
        self.transform = None
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])

        self.image_list = natsort.natsorted(Image_dir_path.iterdir())
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
   
        image_path = str(self.image_list[idx])

        image_raw = cv2.imread(image_path)
        ##
        
        
        crop_img , draw_num , bound_line = self.extract_draw_num(image_raw)
        
        if self.transform:
            crop_img = self.transform(crop_img).unsqueeze(0)
        crop_img = crop_img.type(torch.float32)
        
        return {'image': crop_img, 
                    'image_raw': image_raw, 
                    'image_name': image_path, 
                    'image_h': crop_img.shape[-3],
                   'image_w': crop_img.shape[-2],
                   'bounds' : bound_line,
                   'draw_num' : draw_num,
                   'page_num' : idx+1 
            }
        
        
    def extract_draw_num(self , image):
        
        draw_num = None
        sorted_stats = get_stats(image)
     
        if len(sorted_stats) < 3:
            return image , None , (0 , 0 , image.shape[1] , image.shape[0])
    
        (left_bound , down_bound , right_bound , upper_bound) = get_bound(sorted_stats)

        if left_bound - 200 < 0 : 
            return image, None , (0 , 0 , image.shape[1] , image.shape[0])

        cvt_img = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)

        graph = cvt_img[1500:1700 , left_bound-200:left_bound+200]
        graph2 = cvt_img[: , left_bound:]

        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel('FSRCNN_x4.pb')
        sr.setModel("fsrcnn",4)
        
        graph = sr.upsample(graph)
        graph2 = sr.upsample(graph2)

        reader = easyocr.Reader(['en'])
        easy_result = reader.readtext(graph, allowlist = 'ABCDEFGHIZKLMNOPRSTUVWXYZ-1234567890')
        easy_result2 = reader.readtext(graph2, allowlist = 'ABCDEFGHIZKLMNOPRSTUVWXYZ-1234567890')


        if draw_num == None:
            for text in easy_result:
                if re.findall(r'\b([A-Z]{1,3}\s{0,}[-]\s{0,}\w{1,3})\b', str(text)):
                    draw_num = text[1]
                
        if draw_num == None:
            for text in easy_result2:
                if re.findall(r'\b([A-Z]{1,3}\s{0,}[-]\s{0,}\w{1,3})\b', str(text)):
                    draw_num = text[1]
                    
                
        horizontal_lists, free_lists  = reader.detect(image)
        horizontal_list, _ = horizontal_lists[0], free_lists[0]
        copy_image = copy.deepcopy(image)
        for bbox in horizontal_list:       
            copy_image[bbox[2]:bbox[3] , bbox[0]:bbox[1]].fill(255) 
             
        crop_img = copy_image[upper_bound:down_bound , right_bound:left_bound]  
        return crop_img , draw_num , (upper_bound , down_bound , right_bound , left_bound)


#template는 이미지내에 존재하는 심볼의 종류이다.
# templateDataset:
#       image : symbol detect를 수행할 이미지
#       template : detect를 수행할 심볼 이미지
#       image_h , image_w : symbol의 높이와 너비
#       thresh : 해당 심볼의 민감도

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

        self.template_path = natsort.natsorted(template_dir_path.iterdir())
        
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
        
        
        
        
        