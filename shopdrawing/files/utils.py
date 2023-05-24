from __future__ import print_function, division
import matplotlib.pyplot as plt
import math
from sklearn.metrics import auc
import numpy as np
import cv2
import os, sys , re
import fitz


int_ = lambda x: int(round(x))


def char2num(draw_num):
    regex = re.compile('[a-zA-Z]')
    for match in regex.finditer(draw_num.split('-')[1]):
        char = match.group()
        if char == 'O': draw_num = draw_num.replace('O' , '0' )
        elif char == 'S': draw_num = draw_num.replace('S' , '5')
        elif char == 'A': draw_num= draw_num.replace('A' , '4')
        elif char == 'I': draw_num = draw_num.replace('I' , '1')
        
    return draw_num

def pdf_to_png(PDF_FILE_PATH):
    doc = fitz.open(PDF_FILE_PATH)
    save_name = PDF_FILE_PATH.split('/')[-1].split('.pdf')[0]
    save_dir = os.path.join('/home/gpuadmin/shopdwg/shopdrawing/data/images' , save_name)
    os.makedirs(save_dir, exist_ok=True)
    
    #(1755, 2482, 3)
    mat = fitz.Matrix(2, 2)
    for i, page in enumerate(doc):
        img = page.get_pixmap(matrix = mat)
        img.save(f"{save_dir}/image_{i}.png")


def IoU( r1, r2 ):
    x11, y11, w1, h1 = r1
    x21, y21, w2, h2 = r2
    x12 = x11 + w1; y12 = y11 + h1
    x22 = x21 + w2; y22 = y21 + h2
    x_overlap = max(0, min(x12,x22) - max(x11,x21) )
    y_overlap = max(0, min(y12,y22) - max(y11,y21) )
    I = 1. * x_overlap * y_overlap
    U = (y12-y11)*(x12-x11) + (y22-y21)*(x22-x21) - I
    J = I/U
    return J


def evaluate_iou( rect_gt, rect_pred ):
    # score of iou
    score = [ IoU(i, j) for i, j in zip(rect_gt, rect_pred) ]
    return score


def compute_score( x, w, h ):
    # score of response strength
    k = np.ones( (h, w) )
    score = cv2.filter2D(x, -1, k)
    score[:, :w//2] = 0
    score[:, math.ceil(-w/2):] = 0
    score[:h//2, :] = 0
    score[math.ceil(-h/2):, :] = 0
    return score


def locate_bbox( a, w, h ):
    row = np.argmax( np.max(a, axis=1) )
    col = np.argmax( np.max(a, axis=0) )
    x = col - 1. * w / 2
    y = row - 1. * h / 2
    return x, y, w, h


def score2curve( score, thres_delta = 0.01 ):
    thres = np.linspace( 0, 1, int(1./thres_delta)+1 )
    success_num = []
    for th in thres:
        success_num.append( np.sum(score >= (th+1e-6)) )
    success_rate = np.array(success_num) / len(score)
    return thres, success_rate


def all_sample_iou( score_list, gt_list):
    num_samples = len(score_list)
    iou_list = []
    for idx in range(num_samples):
        score, image_gt = score_list[idx], gt_list[idx]
        w, h = image_gt[2:]
        pred_rect = locate_bbox( score, w, h )
        iou = IoU( image_gt, pred_rect )
        iou_list.append( iou )
    return iou_list


def plot_success_curve( iou_score, title='' ):
    thres, success_rate = score2curve( iou_score, thres_delta = 0.05 )
    auc_ = np.mean( success_rate[:-1] ) # this is same auc protocol as used in previous template matching papers #auc_ = auc( thres, success_rate ) # this is the actual auc
    plt.figure()
    plt.grid(True)
    plt.xticks(np.linspace(0,1,11))
    plt.yticks(np.linspace(0,1,11))
    plt.ylim(0, 1)
    plt.title(title + 'auc={}'.format(auc_))
    plt.plot( thres, success_rate )
    plt.show()
    
    

    
    

