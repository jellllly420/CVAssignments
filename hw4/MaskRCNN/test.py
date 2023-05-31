from tkinter import Label
import utils
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNN
from dataset import SingleShapeDataset
from utils import plot_save_output
import torch
import numpy as np
import torch.utils.data




# the outputs includes: 'boxes', 'labels', 'masks', 'scores'

def compute_detection_ap(output_list, gt_labels_list, iou_threshold=0.5):
    def iou(box1, box2):
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
        x_overlap = max(0, min(xmax1, xmax2) - max(xmin1, xmin2))
        y_overlap = max(0, min(ymax1, ymax2) - max(ymin1, ymin2))
        intersection = x_overlap * y_overlap
        union = (xmax1 - xmin1) * (ymax1 - ymin1) + (xmax2 - xmin2) * (ymax2 - ymin2) - intersection
        return float(intersection) / union

    predictions_ = []
    for i in range(len(gt_labels_list)):
        for box, label, score in zip(output_list[i]['boxes'], output_list[i]['labels'], output_list[i]['scores']):
            predictions_.append(torch.concatenate([torch.Tensor([i]), torch.Tensor([score]), torch.Tensor([label]), box]))
    predictions_ = torch.stack(predictions_)
    predictions_ = predictions_[predictions_[:, 1].argsort(descending = True)]

    aps = []
    for i in range(1, 4):
        num_gt = 0
        for j in range(len(gt_labels_list)):
            if gt_labels_list[j]['labels'][0] == i:
                num_gt += 1
        if num_gt == 0:
            continue
        
        predictions = predictions_[predictions_[:, 2] == i]
        flags = [False for _ in range(len(gt_labels_list))]
        tp = 0
        precisions = []
        recalls = []
        for rank, prediction in enumerate(predictions):
            if prediction[2] == i and \
                gt_labels_list[int(prediction[0])]['labels'][0] == i and \
                flags[int(prediction[0])] == False and \
                iou(prediction[3:], gt_labels_list[int(prediction[0])]['boxes'][0]) > iou_threshold:
                flags[int(prediction[0])] = True
                tp += 1
            precisions.append(tp / (rank + 1))
            recalls.append(tp / num_gt)

        interpolates = []
        for j in range(11):
            interpolate = 0
            for precision, recall in zip(precisions, recalls):
                if recall >= j / 10:
                    interpolate = max(interpolate, precision)
            interpolates.append(interpolate)
        aps.append(sum(interpolates) / 11)
        '''
        print(num_gt)
        print(precisions)
        print(recalls)
        print(interpolates)
        print(aps)
        print()
        '''
    mAP_detection = sum(aps) / len(aps)
        
    return mAP_detection



def compute_segmentation_ap(output_list, gt_labels_list, iou_threshold=0.5):
    def iou(mask1, mask2):
        mask1 = torch.reshape(mask1, mask2.shape)
        mask1 = torch.where(mask1 > 0.5, 1, 0)
        intersection = torch.sum(torch.logical_and(mask1, mask2))
        union = torch.sum(torch.logical_or(mask1, mask2))
        #print(float(intersection) / union)
        return float(intersection) / union

    predictions_ = []
    for i in range(len(gt_labels_list)):
        for mask, label, score in zip(output_list[i]['masks'], output_list[i]['labels'], output_list[i]['scores']):
            predictions_.append(torch.concatenate([torch.Tensor([i]), torch.Tensor([score]), torch.Tensor([label]), torch.flatten(mask)]))
    predictions_ = torch.stack(predictions_)
    predictions_ = predictions_[predictions_[:, 1].argsort(descending = True)]

    aps = []
    for i in range(1, 4):
        num_gt = 0
        for j in range(len(gt_labels_list)):
            if gt_labels_list[j]['labels'][0] == i:
                num_gt += 1
        if num_gt == 0:
            continue
        
        predictions = predictions_[predictions_[:, 2] == i]
        flags = [False for _ in range(len(gt_labels_list))]
        tp = 0
        precisions = []
        recalls = []
        for rank, prediction in enumerate(predictions):
            if gt_labels_list[int(prediction[0])]['labels'][0] == i and \
                flags[int(prediction[0])] == False and \
                iou(prediction[3:], gt_labels_list[int(prediction[0])]['masks'][0]) > iou_threshold:
                flags[int(prediction[0])] = True
                tp += 1
            precisions.append(tp / (rank + 1))
            recalls.append(tp / num_gt)

        interpolates = []
        for j in range(11):
            interpolate = 0
            for precision, recall in zip(precisions, recalls):
                if recall >= j / 10:
                    interpolate = max(interpolate, precision)
            interpolates.append(interpolate)
        aps.append(sum(interpolates) / 11)
        '''
        print(num_gt)
        print(precisions)
        print(recalls)
        print(interpolates)
        print(aps)
        print()
        '''
    mAP_segmentation = sum(aps) / len(aps)



    return mAP_segmentation







dataset_test = SingleShapeDataset(100)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)
 

num_classes = 4
 
# get the model using the helper function
model = utils.get_instance_segmentation_model(num_classes).double()

device = torch.device('cpu')


# replace the 'cpu' to 'cuda' if you have a gpu
model.load_state_dict(torch.load(r'xxx/MaskRCNN/results/maskrcnn_23.pth',map_location='cpu'))



model.eval()
path = "results/" 
# # save visual results
for i in range(10):
    imgs, labels = dataset_test[i]
    output = model([imgs])
    plot_save_output(path+str(i)+"_result.png", imgs, output[0])


# compute AP
gt_labels_list = []
output_label_list = []
with torch.no_grad():
    for i in range(100):
        print(i)
        imgs, labels = dataset_test[i]
        gt_labels_list.append(labels)
        output = model([imgs])
        output_label_list.append(output[0])

mAP_detection = compute_detection_ap(output_label_list, gt_labels_list)
mAP_segmentation = compute_segmentation_ap(output_label_list, gt_labels_list)


np.savetxt(path+"mAP.txt",np.asarray([mAP_detection, mAP_segmentation]))

