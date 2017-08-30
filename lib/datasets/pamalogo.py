# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.pamalogo
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess

class pamalogo(datasets.imdb):
    def __init__(self, image_set, devkit_path):
        datasets.imdb.__init__(self, image_set)
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._data_path = os.path.join(self._devkit_path, 'data')
        self._classes = ('__background__', # always index 0
                         # number of pamalogo classes: 130
                         'Academy_Awards','Adidas-Pict','Adidas-Text','AFD','Aldi','Allianz-Pict',
    'Allianz-Text','Amazon','Amazon-Pict','Amazon-Text','Amnesty_internatinal','Apple',
    'Atletico_Madrid','Audi-Pict','Audi-Text','Bayer_Leverkusen','Berlinale','BMW','Borussia_Dortmund',
    'Borussia_Monchengladbach','Brasilien','Bundesliga','Bundeswappen','Bundnis','Burger_king','CDU','CSU','China',
    'CocaCola','Deutsche_Bahn','Deutsche_Bank-Pict','Deutsche_Bank-Text','Deutsche_Post_AG-Pict',
    'Deutsche_Post_AG-Text','Deutsche_Telekom-Pict','Deutsche_Telekom-Text','DFB','Die_Linke','eBay',
    'Eurovision_Song_Contest','EZB','Facebook-Pict','Facebook-Text','FC_Barcelona',
    'FC_Bayern_Munchen','FDP','Ferrari-Pict','Ferrari-Text','Fubball_Weltmeisterschaft','Ford','GDL','GEW',
    'German_Flag','Germany','Google','Great_Seal_us','Greenpeace','IAAF','IG_Metall','IKEA','Intel','IOC','IPC','IS',
    'Lego','Lufthansa-Pict','Lufthansa-Text','MasterCard-Pict','MasterCard-Text','McDonalds-Pict',
    'McDonalds-Text','Mercedes_Benz-Pict','Mercedes_Benz-Text','Microsoft-Pict','Microsoft-Text',
    'Monster_Energy-Pict','Monster_Energy-Text','NASA','NATO','Nigeria','Nike-Pict','Nike-Text',
    'Nordkorea','NPD','Olympische_Sommerspiele','Opel-Pict','Opel-Text','Paralympische_Sommerspiele',
    'PayPal-Pict','PayPal-Text','Pepsi','Piraten','Polizei','Quick-Pict','Quick-Text','Real_Madrid','Red_Bull-Pict',
    'Red_Bull-Text','Reebok-Pict','Reebok-Text','Samsung','Seal_us_president','Siemens','Sofortuberweisung-Pict',
    'Sofortuberweisung-Text','Sony','SPD','Starbucks','Sudkorea','Syrien','Twitter-Pict','Twitter-Text',
    'UEFA','UN','Under_Armour','United_Kingdom','US_flag','Vattenfall-Pict',
    'Vattenfall-Text','ver_di','VfL_Wolfsburg','Visa','Vodafone-Pict','Vodafone-Text','Volkswagen',
    'Walt_Disney_Pictures','WhatsApp-Pict','WhatsApp-Text','Zalando-Pict','Zalando-Text')
        
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = ['.jpg', '.png','.JPG', '.jpeg']
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # Specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert os.path.exists(self._devkit_path), \
                'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        for ext in self._image_ext:
            image_path = os.path.join(self._data_path, 'Images',
                                  index + ext)
            if os.path.exists(image_path):
                break
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_pamalogo_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
            print len(roidb)
        with open(cache_file, 'wb') as fid:
                cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self._devkit_path,
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['all_boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_IJCV_roidb(self):
        """
        eturn the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                '{:s}_selective_search_IJCV_top_{:d}_roidb.pkl'.
                format(self.name, self.config['top_k']))

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_IJCV_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_IJCV_roidb(self, gt_roidb):
        IJCV_path = os.path.abspath(os.path.join(self.cache_path, '..',
                                                 'selective_search_IJCV_data',
                                                 self.name))
        assert os.path.exists(IJCV_path), \
               'Selective search IJCV data not found at: {}'.format(IJCV_path)

        top_k = self.config['top_k']
        box_list = []
        for i in xrange(self.num_images):
            filename = os.path.join(IJCV_path, self.image_index[i] + '.mat')
            raw_data = sio.loadmat(filename)
            box_list.append((raw_data['boxes'][:top_k, :]-1).astype(np.uint16))

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pamalogo_annotation(self, index):
        """
        Load image and bounding boxes info from txt files of pamalogoPerson.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        # print 'Loading: {}'.format(filename)
        with open(filename) as f:
                data = f.read()
        import re
        objs = re.findall('\(\d+, \d+\)[\s\-]+\(\d+, \d+\)', data)
        
        num_objs = len(objs)
    
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            coor = re.findall('\d+', obj)
            x1 = float(coor[0])
            y1 = float(coor[1])
            x2 = float(coor[2])
            y2 = float(coor[3])
            #cls = self._class_to_ind['person']
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
    
        overlaps = scipy.sparse.csr_matrix(overlaps)
    
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def _write_pamalogo_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # VOCdevkit/results/comp4-44503_det_test_aeroplane.txt
        path = os.path.join(self._devkit_path, 'results', self.name, comp_id + '_')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} results file'.format(cls)
            filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        return comp_id

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'setenv(\'LC_ALL\',\'C\'); voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)
    
    def _cls_evaldet(self, cls, ids, cls_det, gt_det, gtids_hash):
        npos = 0
        cls_gt=[]
        for i in range(0, len(gt_det['gtids'])):
            clsinds = [k for k,this_cls in enumerate(gt_det['recs'][0][i]['objects']['class'][0]) if str(this_cls[0]) == cls]
            this_gt = {}
            this_gt['BB'] = [gt_det['recs'][0][i]['objects']['bbox'][0][k][0] for k in clsinds]
            this_gt['diff'] = [gt_det['recs'][0][i]['objects']['difficult'][0][k][0][0] for k in clsinds]
            this_gt['det'] = np.zeros((len(clsinds),1))
            cls_gt.append(this_gt)
            npos = npos + this_gt['diff'].count(0)
    
        nd = len(cls_det)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
    
        for d in range(0, nd):
            i = gtids_hash[ids[d]]
            #TODO: Handle unrecognized ID
        
            bb = [round(t,1)+1 for t in cls_det[d][0:4]]
            ovmax = -1
            for j in range(0,len(cls_gt[i]['BB'])):
                bbgt = cls_gt[i]['BB'][j]
            
            bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
            iw = bi[2] - bi[0] + 1
            ih = bi[3] - bi[1] + 1
            if iw>0 and ih>0:
                ua = (bb[2] - bb[0] + 1)*(bb[3] - bb[1]+1)+(bbgt[2]-bbgt[0]+1)*(bbgt[3]-bbgt[1]+1) - iw*ih
                ov = iw*ih/ua
                if ov>ovmax:
                    ovmax = ov
                    jmax = j
        
            if ovmax>=0.5: #TODO: Parameter to change threshold
                if not cls_gt[i]['diff'][jmax]:
                    if not cls_gt[i]['det'][jmax]:
                        tp[d]=1
                        cls_gt[i]['det'][jmax]=1
                    else:
                        fp[d]=1
            else:
                fp[d]=1
    
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp/npos
        prec = [tp[i]/(tp[i]+fp[i]) for i in range(0,len(fp))]
        mrec = [0]
        mrec.extend(rec)
        mrec.append(1)
        mpre = [0]
        mpre.extend(prec)
        mpre.append(0)
        for i in range(len(mpre)-2,-1,-1):
            mpre[i] = max(mpre[i],mpre[i+1])
        
        ap = sum([(mrec[i+1] - mrec[i])*(mpre[i+1]) for i in range(len(mrec)-1) if not mrec[i]==mrec[i+1]])
        res={}
        res['prec']=prec
        res['rec']=rec
        res['ap']=ap
    
        return res
    
    def _do_python_eval(self, all_boxes, output_dir):
        #gt_path = os.path.join(self._devkit_path, 'local', 'VOC' + self._year,
                            #self._image_set+'_anno.mat')
        gt_path = os.path.join(self._devkit_path, 'test.mat')
        gt_det = sio.loadmat(gt_path) #The gtids, recs mat file
        
        gtid_hash={}
        for i,id in enumerate(gt_det['gtids']):
            gtid_hash[str(id[0][0])]=i
    
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Computing results for {}'.format(cls)
            cls_det = all_boxes[cls_ind][:]
        
            ids = []
            new_cls_det=[]
            for i in range(0,len(cls_det)):
                for bbox in cls_det[i]:
                    ids.append(str(self.image_index[i]))
                    new_cls_det.append(bbox)
            scores = np.array([-1*round(b[-1],3) for b in new_cls_det])
            si = np.lexsort((ids,scores))
            scores = scores[si]
            new_cls_det = np.array(new_cls_det)
            ids = np.array(ids)
            new_cls_det = new_cls_det[si]
            ids = ids[si]
    
            cls_res = self._cls_evaldet(cls, ids, new_cls_det, gt_det, gtid_hash)
            cls_outpath = os.path.join(output_dir,cls+'.pkl')
            with open(cls_outpath,'wb') as output:
                cPickle.dump(cls_res,output,cPickle.HIGHEST_PROTOCOL)
    
            print 'AP: {:4f}'.format(cls_res['ap'])

    def evaluate_detections(self, all_boxes, output_dir):
        #comp_id = self._write_pamalogo_results_file(all_boxes)
        #self._do_matlab_eval(comp_id, output_dir)
        self._do_python_eval(all_boxes,output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.pamalogo('train', '')
    res = d.roidb
    from IPython import embed; embed()

