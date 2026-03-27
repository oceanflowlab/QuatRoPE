import numpy as np
import json
import sys
sys.path.append('.')
import torch
import random
from tqdm import tqdm
from collections import defaultdict
import argparse
from utils.box_utils import get_box3d_min_max, box3d_iou, construct_bbox_corners
from prompts.prompts import grounding_prompt
import string


parser = argparse.ArgumentParser()

parser.add_argument('--segmentor', required=True, type=str)
parser.add_argument('--version', type=str, default='')
parser.add_argument('--train_iou_thres', type=float, default=0.75)
parser.add_argument('--max_obj_num', type=int, default=150)
args = parser.parse_args()

segmentor = args.segmentor
version = args.version

banned_heads = ['The object is the location', 'The object is the color', 'The object is not', 'The object is the size', 'The object is the shape']
banned_conts = ['another']
banned_tails = ['on', 'attached to', 'made of', 'in front of', 'closest to', 'under', 'above', 'behind']

for split in ["val"]:
    count = [0] * args.max_obj_num
    annos = json.load(open(f"annotations/scanqa/ScanQA_v1.0_sub_{split}.json", "r"))
    annos = sorted(annos, key=lambda p: f"{p['scene_id']}_{int(p['object_ids'][0]):03}")
    new_annos = []

    instance_attribute_file = f"annotations/scannet_{segmentor}_{split}_attributes{version}.pt"
    scannet_attribute_file = f"annotations/scannet_{split}_attributes.pt"
    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    for i, anno in tqdm(enumerate(annos)):
        scene_id = anno['scene_id']
        obj_id = int(anno['object_ids'][0])
        desc = anno['question']

        skip = False
        for banned_head in banned_heads:
            if desc.startswith(banned_head):
                skip = True
        for banned_cont in banned_conts:
            if banned_cont in desc:
                skip = True
        for banned_tail in banned_tails:
            if desc.endswith(banned_tail + '.'):
                skip = True
                break
        if skip:
            continue

        if desc[-1] in string.punctuation:
            desc = desc[:-1]
        prompt = random.choice(grounding_prompt).replace('<description>', desc)
        
        if scene_id not in instance_attrs:
            continue
        instance_locs = instance_attrs[scene_id]["locs"]
        scannet_locs = scannet_attrs[scene_id]["locs"]
        instance_num = instance_locs.shape[0]
        max_iou, max_id = -1, -1
        for pred_id in range(instance_num):
            pred_locs = instance_locs[pred_id].tolist()
            gt_locs = scannet_locs[obj_id].tolist()
            pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
            gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
            iou = box3d_iou(pred_corners, gt_corners)
            if iou > max_iou:
                max_iou = iou
                max_id = pred_id
        count[max_id] += 1
        
        new_annos.append({
            "scene_id": scene_id,
            "obj_id": obj_id,
            "ref_captions": [f"<OBJ{max_id:03}>."],
            "prompt": prompt
        })

    print(len(new_annos))
    print(count)

    with open(f"annotations/asr_{segmentor}_{split}{version}.json", "w") as f:
        json.dump(new_annos, f, indent=4)

