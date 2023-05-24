from torchvision import models , transforms , utils

from files.feature_extract import *
from files.nms_plot import *
 
from files.DataLoader import ImageDataset
from files.DataLoader import TemplateDataset

import json ,easyocr
from collections import OrderedDict

def symbol_recog(boxes , indices , image_raw):
    reader = easyocr.Reader(['en'])
    object_list = []
    draw_num_list = [] #에러를 줄이기 위한 방법
    result_list = []
    for index , (box , indice) in enumerate(zip(boxes , indices)):
        
        object_dict = OrderedDict()

        #좌측 상단 좌표 
        object_dict['x'] = box[0][0]
        object_dict['y'] = box[0][1]
        object_dict['w'] = box[1][1] - box[0][1]
        object_dict['h'] = box[1][0] - box[0][0]
        crop_image = image_raw[box[0][1]:box[1][1] , box[0][0]:box[1][0]]
                
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel('FSRCNN_x4.pb')
        sr.setModel("fsrcnn",4)
        
        crop_image = sr.upsample(crop_image)
        result = reader.readtext(crop_image , allowlist = 'ABCDEFGHIZKLMNOPRSTUVWXYZ-1234567890')
        result_list.append(result)
        # location , value , confidence = result[0][0] , result[0][1] , result[0][2]
        if len(result):
            draw_num_list.append(result[0][1])
            object_dict['data'] = result[0][1]
        else : 
            object_dict['data'] = 'None'
            
        if re.findall(r'\b([A-Z]{1,3}[-]\d{1,4})\b', object_dict['data']):
            if len(object_dict['data'].split('-')[1]) > 3:
                object_dict['data'] = object_dict['data'].split('-')[0] + object_dict['data'].split('-')[1][:3]
            object_dict['type']  =  "drawing_no"     
        else:
            object_dict['type'] = 'None'
            
        
        object_list.append(object_dict)
        
    return object_list , result_list



def find(image_dir , template_dir , save_name , image_sample =False):
    
    save_dir = os.path.join('./results/' , save_name) 
    
    # 도면 이미지를 데이터 로더 클래스로 가져온다.
    # 이미지 하나하나 처리, 이미지내에 도면번호를 추출하고 템플릿이 존재 하는지 확인하다. 
    # 도면번호는 imageDataset class 내부의 함수로 처리한다. DataLoader.py 에 정의되어있음    
    images = ImageDataset(Path(image_dir)) 
    drawing_list = []
    results = []
    for index , image in enumerate(images):
        
        # josn 파일에 들어갈 파일 생성 
        drawing_dict = OrderedDict()
        drawing_dict['page_index'] =image['page_num']
        drawing_dict['size'] = {'width' : image['image_raw'].shape[1] , 'height' :  image['image_raw'].shape[0]}

        drawing_dict['bound'] = {'top' : image['bounds'][0] , 'bottom' : image['bounds'][1],
                                'left' : image['bounds'][2] , 'right' : image['bounds'][3]
                                }

        
        draw_num = image['draw_num']  
        if draw_num == None :
            drawing_dict['drawing_no'] =  'None'
            if image_sample:
                os.makedirs(os.path.join(save_dir , 'samples') ,exist_ok= True)
                plt.imsave(os.path.join( save_dir , 'samples', 'image_'+str(image['page_num'])+'.png') , image['image_raw'])
                drawing_list.append(drawing_dict)
            continue
        # 도면번호가 영어로 인식된 경우 일부 휴리스틱하게 처리 
        draw_num = char2num(draw_num)
        drawing_dict['drawing_no'] =  draw_num
                
        templates = TemplateDataset(
                        Path(template_dir), 
                        image['image'],  
                        thresh_csv='./template/thresh_template_example.csv'
                        )
        
        
        # 템플릿 이미지와 도면 이미지의 특징을 추출하는 모델, 여기서는 vgg 모델을 사용함
        # run_multi _sample는 발견한 template 마다의 점수와 심볼의 위치를 반환한다.
        # nms_multi 위 결과를 통해 template의 점수증 가장 큰 값을 레이블로 하고 위치 정보를 이용해 box 값을 준다.
        # plot_result_multi 에서는 boxes 위치에 시각화를 위한 경계선을 그려준다.
        
        model = CreateModel(model=models.vgg19(pretrained=True).features, alpha=25, use_cuda=True)
        scores, w_array, h_array, thresh_list = run_multi_sample(model, templates)

        boxes, indices = nms_multi(scores, w_array, h_array, thresh_list)
        # box위치는 전처리된 이미지 내에서 진행되었기 떄문에 그에 따라 위치를 조정해준다.
        # for box in boxes:
        #     box[0][1] +=  image['bounds'][0]
        #     box[0][0] += image['bounds'][2]
        #     box[1][1] +=  image['bounds'][0]
        #     box[1][0] += image['bounds'][2]
        
        # symbol detect가 수행된 결과를 저장한다.
        
        if image_sample:
            visual_img = plot_result_multi(image["image_raw"], boxes, indices, show=False)
            plt.imsave(os.path.join( save_dir , 'samples', 'image_'+str(image['page_num'])+'.png') , visual_img)


        object_list ,result_list = symbol_recog(boxes , indices , image['image_raw'] )
        drawing_dict['objects'] = object_list
        results.append(result_list)
        drawing_list.append(drawing_dict)
        
    return drawing_list , results

    