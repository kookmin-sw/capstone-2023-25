from finder import *
from datetime import datetime


image_dir = '/home/gpuadmin/shopdwg/shopdrawing/data/images/A3_project'
template_dir = '/home/gpuadmin/shopdwg/shopdrawing/template/template_example'
#pdf_to_png('/home/gpuasdmin/shopdwg/shopdrawing/data/pdf/건축도면(A3).pdf')

version = 1

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
        


def run(template_dir , project_name , save_sample = False):
    
    image_dir = os.path.join('./data' , 'images' , str(project_name))
    pdf_name = './data/pdf/'+ str(project_name) +'.pdf'
    
    if not os.path.exists(pdf_name):
       raise Exception("존재하지 않는 pdf입니다. 프로젝트 명을 확인해주세요") 
    if not os.path.exists(image_dir):
        pdf_to_png(pdf_name)
        
    file_data = OrderedDict()
    file_data = {
        "version": version,
        "meta": {
            "filename": str(project_name) +'.pdf',
            "size": os.path.getsize(pdf_name),  # Replace with the actual byte size of the file
            "processed_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
    }
    
    drawing_list , results = find(image_dir , template_dir , project_name , save_sample)
    file_data['drawings'] = drawing_list
    
    os.makedirs(os.path.join('./results' ,project_name) , exist_ok=True)
    os.makedirs(os.path.join('./results' ,project_name , 'samples') , exist_ok=True)
    os.makedirs(os.path.join('./results' ,project_name , 'drawing_json') , exist_ok=True)
    
    save_name = os.path.join('./results' ,project_name , 'drawing_json' , 'draw_info.json') 
    with open(save_name , 'w' , encoding = 'utf-8') as make_file:
        json.dump(file_data , make_file , ensure_ascii = False , indent = "\t" , cls = NpEncoder)
    
    print("end")
    return drawing_list , results
    
if __name__ == '__main__':
    drawing_list , results = run(template_dir , 'example' , True)

    