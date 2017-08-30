#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

# added by rfm
#import dlib
from skimage import io
from os import listdir

#'''
CLASSES = ('__background__', # always index 0
            # number of pamalogo classes: 20 + 1 bg
            'Adidas-Pict','Adidas-Text','Aldi','Allianz-Pict','Allianz-Text','Amazon','Apple',
            'Atletico_Madrid','Audi-Pict','Audi-Text','BMW','Burger_king','CocaCola','eBay',
            'Facebook-Pict','Facebook-Text','FC_Barcelona','FC_Bayern_Munchen','Ferrari-Pict','Ferrari-Text')
#'''

NETS = {'vgg16': ('VGG16', 'VGG16_faster_rcnn_ld20cls_final.caffemodel'),
        'zf': ('ZF', 'ZF_faster_rcnn_ld20clsfinal.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
        
    l_bboxes = []    
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        print ('Det: (x_min,y_min,W,H) = ({},{},{},{}), class_name = {:s}, score = {:.3f}').format(
                int(bbox[0]),int(bbox[1]),int(bbox[2]-bbox[0]),int(bbox[3]-bbox[1]),class_name,score)
        cv2.rectangle(im, (bbox[0], bbox[3]),(bbox[2],bbox[1]), (0,255,0),2)      
        cv2.putText(im,'{:s}:{:.3f}'.format(class_name, score),
                (int(bbox[0]), int(bbox[1]) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
        
        l_bboxes.append({'x_min':int(bbox[0]),'y_min':int(bbox[1]),'x_max':bbox[2],'y_max':bbox[3],'cls':class_name,'score':score})
    
    return l_bboxes

def demo(net, im_pn):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(im_pn)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    image_name = im_pn.split('/')[-1].split('.')[0]
    im_txt = ('img: {}').format(image_name)
    
    l_bboxes_ALL = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
            continue
        
        l_bboxes = vis_detections(im, cls, dets, thresh=CONF_THRESH)
        l_bboxes_ALL.extend(l_bboxes)
        im_txt += (', class: {}').format(cls)
    
    out_dir = os.path.join(cfg.DATA_DIR, 'demo_out')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    im_out_fn = os.path.join(out_dir, im_pn.split("/")[-1])
        
    if len(l_bboxes_ALL) > 0:
        #cv2.imshow(im_txt,im)        
        cv2.imwrite(im_out_fn,im)
    
    return l_bboxes_ALL

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],'faster_rcnn_end2end', 'test_ld.20cls.prototxt')    
    caffemodel = os.path.join(cfg.DATA_DIR, 'logo_models', NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    set_pn = os.path.join(cfg.DATA_DIR, 'VOCdevkit2007/VOC2007/ImageSets/Main/test.txt')
    set_im_dir = os.path.join(cfg.DATA_DIR, 'VOCdevkit2007/VOC2007/JPEGImages')
    
    im_pns = []
    with open(set_pn) as in_file:
        for row in in_file:
            im_fn = row.split("\r")[0].split("\n")[0]+".jpg"
            #print "im_fn: ", im_fn
            im_pn = os.path.join(set_im_dir, im_fn)
            im_pns.append(im_pn)
        in_file.close()
        
    for im_pn in im_pns:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {}'.format(im_pn)
        l_bboxes_ALL = demo(net, im_pn)

    cv2.waitKey(0)
