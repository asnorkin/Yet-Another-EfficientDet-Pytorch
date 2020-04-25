import cv2 as cv
import glob
import  json
import numpy as np
import os
import os.path as osp
from argparse import ArgumentParser
from shutil import rmtree


def config_data_sources(args):
    names = [
        'atletico_autofollow',
        'atm_lev_autofollow',
        'cag_ver_autofollow',
        'inter_autofollow',
        'inter_panoramic',
        'lec_juv_autofollow',
        'lec_juv_panoramic',
        'mci-tot_autofollow',
        'mci_autofollow',
        'real_barca_autofollow',
        'roma_autofollow',
        'roma_crotone_autofollow'
    ]

    return [
        {
            'images_dir': osp.join(args.data_dir, 'images', name),
            'labels_dir': osp.join(args.data_dir, 'player_labels_extended', name),
            'prefix': name + '_',
        }
        for name in names
    ]


def init_anno():
    categories = [
        {
            'id': 1,
            'name': 'central_referee',
            'supercategory': 'person',
        },
        {
            'id': 2,
            'name': 'ball',
            'supercategory': 'sports',
        },
        {
            'id': 3,
            'name': 'activeplayer',
            'supercategory': 'person',
        },
        {
            'id': 4,
            'name': 'nonplayer',
            'supercategory': 'person',
        },
        {
            'id': 5,
            'name': 'upper_referee',
            'supercategory': 'person',
        },
        {
            'id': 6,
            'name': 'lower_referee',
            'supercategory': 'person',
        },
        {
            'id': 7,
            'name': 'goalkeeper_1',
            'supercategory': 'person',
        },
        {
            'id': 8,
            'name': 'goalkeeper_2',
            'supercategory': 'person',
        },
        {
            'id': 9,
            'name': 'team_1',
            'supercategory': 'person',
        },
        {
            'id': 10,
            'name': 'team_2',
            'supercategory': 'person',
        },
    ]

    anno = {
        'images': [],
        'annotations': [],
        'categories': categories,
    }

    return anno


def config():
    ap = ArgumentParser()

    ap.add_argument('--data_dir', default='../footballobjectdetection/data')
    ap.add_argument('--dataset_dir', default='datasets/player_detection/')
    ap.add_argument('--min_overlap', type=int, default=40)
    ap.add_argument('--n_splits', type=int, default=None)
    ap.add_argument('--val_size', type=float, default=0.1)

    args = ap.parse_args()
    return args


def read(source):
    image_prefix = osp.join(source['images_dir'], source['prefix'])
    image_files = glob.glob(image_prefix + '*.jpg')

    objects_count = 0
    image_indices = list(map(lambda x: int(x[len(image_prefix):].split('.')[0]), image_files))
    for index in image_indices:
        image_file = '{}{}.jpg'.format(image_prefix, index)
        image = cv.imread(image_file)

        label_file = osp.join(source['labels_dir'], source['prefix'] + str(index) + '.txt')
        if osp.exists(label_file):
            label = []
            with open(label_file, 'r') as inpf:
                for line in inpf:
                    line = line.strip().split()
                    label.append([int(line[0])] + list(map(float, line[1:])))

            yield image, label, index
            objects_count += 1

    print('Read {} objects with prefix {}'.format(objects_count, source['prefix']))


def split(full_image, full_label, n_splits=None, min_overlap=40):
    H, W = full_image.shape[:2]

    def overlap(n):
        return int(np.floor((n * H - W) / (n - 1)))

    if n_splits is None:
        n_splits = 2
        while overlap(n_splits) < min_overlap:
            n_splits += 1

    split.overlap = overlap(n_splits)
    split.n_splits = n_splits

    shift = H - split.overlap
    for split_index in range(1, n_splits + 1):
        x, w = (split_index - 1) * shift, H
        image_split = full_image[:, x: x + w]
        if full_label is None:
            yield image_split, None, split_index, x
            continue

        x_from, x_to, x_w = x / W, (x + w) / W, w / W
        label_split = []
        for clid, xc, yc, w, h in full_label:
            if x_from <= xc <= x_to:
                label_split.append([clid, (xc - x_from) / x_w, yc, w / x_w, h])

        yield image_split, label_split, split_index, x


def main(args):
    if osp.exists(args.dataset_dir):
        rmtree(args.dataset_dir)

    os.makedirs(osp.join(args.dataset_dir, 'train'))
    os.makedirs(osp.join(args.dataset_dir, 'val'))
    os.makedirs(osp.join(args.dataset_dir, 'annotations'))

    result = {
        'train': init_anno(),
        'val': init_anno(),
    }

    match_team_id_base = dict()

    image_id, label_id = -1, -1
    data_sources = config_data_sources(args)
    for source_index, source in enumerate(data_sources):
        match = source['prefix'][:-1].rpartition('_')[0]
        if match not in match_team_id_base:
            match_team_id_base[match] = len(match_team_id_base) * len(result['train']['categories'])

        for full_image, full_label, full_index in read(source):
            val = np.random.binomial(1, args.val_size, size=1).astype(np.bool)[0]
            key = 'val' if val else 'train'
            clids = [lbl[0] for lbl in full_label]
            old = len(clids) > 0 and max(clids) < 8
            assert not old, 'Old label: {}'.format(full_index)

            for split_image, split_label, split_index, _split_x in split(full_image, full_label, args.n_splits, args.min_overlap):
                file_name = '{}{}_{}.jpg'.format(source['prefix'], full_index, split_index)
                image_file = osp.join(args.dataset_dir, key,  file_name)
                cv.imwrite(image_file, split_image)

                imh, imw = split_image.shape[:2]
                image_id += 1
                result[key]['images'].append({
                    'file_name': file_name,
                    'height': imh,
                    'width': imw,
                    'id': image_id,
                })

                for clid, xc, yc, w, h in split_label:
                    if clid == 2 or clid == 1:
                        continue

                    team_id = match_team_id_base[match] + clid
                    clid = 0

                    x, y = int(np.round((xc - w / 2) * imw)), int(np.round((yc - h / 2) * imh))
                    w, h = int(np.round(w * imw)), int(np.round(h * imh))

                    label_id += 1
                    result[key]['annotations'].append({
                        'iscrowd': 0,
                        'image_id': image_id,
                        'bbox': [x, y, w, h],
                        'area': w * h,
                        'category_id': clid + 1,
                        'team_id': team_id,
                        'id': label_id,
                    })

    for key in ['train', 'val']:
        result[key]['info'] = {'n_ids': len(match_team_id_base) * len(result[key]['categories'])}

    with open(osp.join(args.dataset_dir, 'annotations', 'instances_train.json'), 'w+') as outf:
        json.dump(result['train'], outf)

    with open(osp.join(args.dataset_dir, 'annotations', 'instances_val.json'), 'w+') as outf:
        json.dump(result['val'], outf)


if __name__ == '__main__':
    args = config()
    main(args)
