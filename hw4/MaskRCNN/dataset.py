import os
from matplotlib import image
import numpy as np
import torch
import torch.utils.data
import random
import cv2
import math 
from utils import plot_save_dataset



class SingleShapeDataset(torch.utils.data.Dataset):
    def __init__(self, size):

        self.w = 128
        self.h = 128
        self.size = size
        print("size",self.size)


    def _draw_shape(self, img, mask, shape_id):
        buffer = 20
        y = random.randint(buffer, self.h - buffer - 1)
        x = random.randint(buffer, self.w - buffer - 1)
        s = random.randint(buffer, self.h//4)
        color = tuple([random.randint(0, 255) for _ in range(3)])

        
        if shape_id == 1:
            cv2.rectangle(mask, (x-s, y-s), (x+s, y+s), 1, -1)
            cv2.rectangle(img, (x-s, y-s), (x+s, y+s), color, -1)

        elif shape_id == 2:
            cv2.circle(mask, (x, y), s, 1, -1)
            cv2.circle(img, (x, y), s, color, -1)

        elif shape_id == 3:
            points = np.array([[(x, y-s),
                            (x-s/math.sin(math.radians(60)), y+s),
                            (x+s/math.sin(math.radians(60)), y+s),
                            ]], dtype=np.int32)
            cv2.fillPoly(mask, points, 1)
            cv2.fillPoly(img, points, color)


    def __getitem__(self, idx):
        np.random.seed(idx)

        n_class = 1
        masks = np.zeros((n_class, self.h, self.w))
        img = np.zeros((self.h, self.w, 3))
        img[...,:] = np.asarray([random.randint(0, 255) for _ in range(3)])[None, None, :]


        obj_ids = np.zeros((n_class)) 

        shape_code = random.randint(1,3)
        self._draw_shape( img, masks[0, :], shape_code)
        obj_ids[0] = shape_code


        boxes = np.zeros((n_class,4))
        pos = np.where(masks[0])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes[0,:] = np.asarray([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        img = torch.tensor(img)
        img = img.permute(2,0,1)


        return img, target

    def __len__(self):
        return self.size




# ----------TODO------------
# Implement ShapeDataset.
# Refer to `SingleShapeDataset` for the shape parameters 
# ----------TODO------------



class ShapeDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.w = 128
        self.h = 128
        self.size = size
        print("size",self.size)

    def _draw_shape(self, img, masks, obj_ids, threshold = 0.2):
        n_shapes = len(obj_ids)
        
        buffer = 20
        for i in range(n_shapes):
            while True:
                y = random.randint(buffer, self.h - buffer - 1)
                x = random.randint(buffer, self.w - buffer - 1)
                s = random.randint(buffer, self.h//4)

                if obj_ids[i] == 1:
                    cv2.rectangle(masks[i], (x-s, y-s), (x+s, y+s), 1, -1)
                elif obj_ids[i] == 2:
                    cv2.circle(masks[i], (x, y), s, 1, -1)
                elif obj_ids[i] == 3:
                    points = np.array([[(x, y-s),
                                    (x-s/math.sin(math.radians(60)), y+s),
                                    (x+s/math.sin(math.radians(60)), y+s),
                                    ]], dtype=np.int32)
                    cv2.fillPoly(masks[i], points, 1)
                
                for j in range(i):
                    iou = np.logical_and(masks[j], masks[i]).sum() / \
                        np.logical_or(masks[j], masks[i]).sum()
                    if iou > threshold:
                        break
                else:        
                    color = tuple([random.randint(0, 255) for _ in range(3)])
                    if obj_ids[i] == 1:
                        cv2.rectangle(img, (x-s, y-s), (x+s, y+s), color, -1)
                    elif obj_ids[i] == 2:
                        cv2.circle(img, (x, y), s, color, -1)
                    elif obj_ids[i] == 3:
                        points = np.array([[(x, y-s),
                                        (x-s/math.sin(math.radians(60)), y+s),
                                        (x+s/math.sin(math.radians(60)), y+s),
                                        ]], dtype=np.int32)
                        cv2.fillPoly(img, points, color)
                    break
                masks[i] = 0
        
        for i in range(n_shapes):
            for j in range(i + 1, n_shapes):
                masks[i][np.where(masks[j])] = 0

    def __getitem__(self, idx):
        np.random.seed(idx)
        n_shapes = random.randint(1,3)

        masks = np.zeros((n_shapes, self.h, self.w))
        img = np.zeros((self.h, self.w, 3))
        img[...,:] = np.asarray([random.randint(0, 255) for _ in range(3)])[None, None, :]

        obj_ids = [random.randint(1, 3) for i in range(n_shapes)]
        self._draw_shape(img, masks, obj_ids)
        obj_ids = np.flip(obj_ids, axis = 0).copy()
        masks = np.flip(masks, axis = 0).copy()

        boxes = np.zeros((n_shapes, 4))
        for i in range(n_shapes):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes[i, :] = np.asarray([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        img = torch.tensor(img)
        img = img.permute(2,0,1)


        return img, target

    def __len__(self):
        return self.size

if __name__ == '__main__':
    # dataset = SingleShapeDataset(10)
    dataset = ShapeDataset(10)
    path = "results/" 

    for i in range(10):
        imgs, labels = dataset[i]
        print(labels)
        plot_save_dataset(path+str(i)+"_data.png", imgs, labels)

