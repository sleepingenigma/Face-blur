import config
import numpy as np
import os
import pandas as pd
import torch
import torchvision

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(
        self,
        root,
        split,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        transform=None,
    ):
        self.dataset = torchvision.datasets.WIDERFace(root=root, split=split)
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        bboxes = self.dataset[index][1]['bbox']
        image = np.array(self.dataset[index][0])
        # print(image.shape)
        size = torch.tensor([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        for i in range(len(bboxes)):
          bboxes[i][0] = bboxes[i][0] + bboxes[i][2]//2
          bboxes[i][1] = bboxes[i][1] + bboxes[i][3]//2
        
        bboxes = bboxes/size
        
        if self.transform:
            image = self.transform(image)/torch.tensor(255.0)

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 5)) for S in self.S]
        k=0
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                if i>=S:
                  k+=1
                  continue
                if j>=S:
                  k+=1
                  continue
                  
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx.item()][anchor_on_scale.item(), i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, tuple(targets)


if __name__ == "__main__":
    anchors = config.ANCHORS

    dataset = YOLODataset(
        root="/content/drive/MyDrive/Yolov3/widerface",
        split='val',
        anchors=anchors,
        transform=config.transforms,
    )
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )

    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    res=0
    for i, (x, y) in enumerate(loader):
      print(x.shape)
    print(res)