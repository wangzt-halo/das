import json
import os
import argparse


def main(args):
    ann_dict_ori = json.load(open(os.path.join(args.root, 'annotations/MuCo-3DHP.json')))
    images = ann_dict_ori['images']
    anns = ann_dict_ori['annotations']
    for ann in anns:
        ann['category_id'] = 1
    images_unaug = [image for image in images if image['file_name'].startswith('unaugmented')]
    images_aug = [image for image in images if image['file_name'].startswith('augmented')]
    print('images_unaug', len(images_unaug))
    print('images_aug', len(images_aug))
    for suffix, imgs in zip(['_unaug', '_aug', '_all'], [images_unaug, images_aug, images]):
        for interv in range(1, 3):
            _images = imgs[::interv]
            _anns = get_selected_anns(_images, anns)
            _ann_dict = {
                'images': _images,
                'annotations': _anns,
            }
            add_categories(_ann_dict)
            ann_name = os.path.join(args.root, 'annotations/train' + suffix + '_interv%d.json' % interv)
            print(ann_name, len(_images))
            with open(ann_name, 'w') as f:
                json.dump(_ann_dict, f)
    return


def get_selected_anns(images, anns):
    image_ids = set([image['id'] for image in images])
    selected_anns = [ann for ann in anns if ann['image_id'] in image_ids]
    return selected_anns


def add_categories(ann_dict):
    category_info = [{
        "supercategory": "person",
        "id": 1,
        "name": "person",
    }]
    ann_dict['categories'] = category_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data/muco')
    parser.add_argument('--interval', type=int, default=1)
    args = parser.parse_args()
    main(args)