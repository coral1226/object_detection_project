import numpy as np
import logging
import pathlib
import json
import cv2
import os
from collections import OrderedDict


class KANADataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False, label_file=None):
        """Dataset for KANA data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            dir_path = self.root / "val"
            anno_path = dir_path / "val_total.json"
        else:
            dir_path = self.root / "train"
            anno_path = dir_path / "train_total.json"

        with open(anno_path) as f:
            img_info_list = json.load(f, object_pairs_hook=OrderedDict)

        self.anno_info = []
        for img_info in img_info_list:
            img_path = os.path.join(dir_path, img_info['filename'])

            ann_info = img_info['ann']
            boxes = ann_info['bboxes']
            labels = ann_info['labels']
            is_difficult = [0 for _ in range(len(labels))]

            self.anno_info.append((img_path,
                                   np.array(boxes, dtype=np.float32),
                                   np.array(labels, dtype=np.int64),
                                   np.array(is_difficult, dtype=np.uint8)))

        self.keep_difficult = keep_difficult

        # classes should be a comma separated list
        self.class_names = (
            'BACKGROUND', "stand", "eavesdrop", "take_object", "weapon", "mask", "cap", "courier_box"
        )
        logging.info("VOC Labels read from file: " + str(self.class_names))

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        img_path, boxes, labels, is_difficult = self.anno_info[index]
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(img_path)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        return image, boxes, labels

    def get_image_and_anno(self, index):
        img_path, boxes, labels, is_difficult = self.anno_info[index]
        image = self._read_image(img_path)
        if self.transform:
            image, _ = self.transform(image)
        return image, (boxes, labels, is_difficult)

    def __len__(self):
        return len(self.anno_info)

    def _read_image(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image



