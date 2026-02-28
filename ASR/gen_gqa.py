import json
import os

if __name__ == '__main__':
    selected_ans = []
    base_pth = './annotations/scanqa'
    with open(os.path.join(base_pth, 'ScanQA_v1.0_val.json'), 'r') as f:
        annos = json.load(f)
    for anno in annos:
        if anno['question'].startswith('What is') and len(anno['object_ids']) == 1 and 'position' not in anno['question']:
            anno['question'] = anno['question'].replace('What is', 'The object is')
            anno['question'] = anno['question'].replace('?', '.')
            selected_ans.append(anno)
    with open(os.path.join(base_pth, 'ScanQA_v1.0_sub_val.json'), 'w') as f:
        json.dump(selected_ans, f, indent=4)

