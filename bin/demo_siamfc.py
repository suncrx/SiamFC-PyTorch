import glob
import os
import pandas as pd
import argparse
import numpy as np
import cv2
import time
import sys

from fire import Fire
from tqdm import tqdm

sys.path.append(os.getcwd())
sys.path.append('../..')


from SiamFC.siamfc import SiamFCTracker

def main(video_dir, gpu_id,  model_path):
    # load videos
    filenames = sorted(glob.glob(os.path.join(video_dir, "frames", "*.jpg")),
           key=lambda x: (os.path.basename(x).split('.')[0]))
    #frames = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in filenames]
    
    # read ground-truth boxes if the file exists.
    gt_box_filepath = os.path.join(video_dir, "gt.csv")
    if os.path.exists(gt_box_filepath):
        gt_bboxes = pd.read_csv(gt_box_filepath, delimiter=', ', #sep='\t|,| ',
                                engine='python')
    else:
        print('ERROR: ', gt_box_filepath, ' does not exist.')
        gt_bboxes = None

       
    # starting tracking
    tracker = SiamFCTracker(model_path, gpu_id)
    for idx, fn in enumerate(filenames):
        # read image and convert color space
        frame = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)
        # get the bbox of the object from the first frame 
        if idx == 0:
            # get the bbox on frame 1
            subdat = gt_bboxes.loc[gt_bboxes['frame']==idx+1]
            bbox = subdat.iloc[0].values[1:5]
            print('ROI:', bbox)
            tracker.init(frame, bbox)
            bbox = (bbox[0]-1, bbox[1]-1,
                    bbox[0]+bbox[2]-1, bbox[1]+bbox[3]-1)
        # track and update the bbox    
        else: 
            bbox = tracker.update(frame)
            #print(bbox)
            
        # bbox xmin ymin xmax ymax
        # draw the bbox of the object
        frame = cv2.rectangle(frame,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])),
                              (0, 255, 0),  2)
        
        # draw the ground-truth bbox of the object
        subdat = gt_bboxes.loc[gt_bboxes['frame']==idx+1]
        if not subdat.empty:            
            gtb = subdat.iloc[0].values[1:5]
            gtb = (gtb[0], gtb[1], gtb[0]+gtb[2], gtb[1]+gtb[3])
            frame = cv2.rectangle(frame,
                                  (int(gtb[0]-1), int(gtb[1]-1)), # 0-index
                                  (int(gtb[2]-1), int(gtb[3]-1)),
                                  (255, 0, 0), 1)
        
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        frame = cv2.putText(frame, str(idx), (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
        cv2.imshow("SiamFC", frame)
        key = cv2.waitKey(10) & 0xff
        if 27==key:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    #Fire(main)
    video_dir = 'D:\\GeoData\\Benchmark\\Video_dataset2014\\dataset\\baseline\\highway'
    model_path = '..\\models\\siamfc_pretrained.pth'
    main(video_dir, 0, model_path)
