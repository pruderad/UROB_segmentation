import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def imshow(img):
    img = (img + 1) / 2  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def get_cls_stats(predict_labels: torch.tensor, labels: torch.tensor, unique_labels: list, ignore_label: int):

    true_posities = []
    true_negatives = []
    false_positives = []
    false_negatives = []

    temp_predict_labels = predict_labels[labels != ignore_label]
    temp_labels = labels.clone()[labels != ignore_label]
    for cls_label in unique_labels:
        
        cls_true_posities = torch.sum(torch.logical_and(temp_labels == cls_label, temp_predict_labels == cls_label)).item()
        cls_true_negatives = torch.sum(torch.logical_and(temp_labels != cls_label, temp_predict_labels != cls_label)).item()
        cls_false_positives = torch.sum(torch.logical_and(temp_labels != cls_label, temp_predict_labels == cls_label)).item()
        cls_false_negatives = torch.sum(torch.logical_and(temp_labels == cls_label, temp_predict_labels != cls_label)).item()

        true_posities.append(cls_true_posities)
        true_negatives.append(cls_true_negatives)
        false_positives.append(cls_false_positives)
        false_negatives.append(cls_false_negatives)

    true_posities = torch.tensor(true_posities)
    true_negatives = torch.tensor(true_negatives)
    false_positives = torch.tensor(false_positives)
    false_negatives = torch.tensor(false_negatives)

    return true_posities, true_negatives, false_positives, false_negatives


def compute_cls_iou(cls_tp: torch.tensor, cls_tn: torch.tensor, cls_fp: torch.tensor, cls_fn: torch.tensor):
    iou = cls_tp / (cls_tp + cls_fp + cls_fn)
    print(iou)
    # iou = true_positive / (true_positive + false_positive + false_negative)
    return iou

def validate(val_dataloader, model, device, criterion, unique_labels: list, visualize: bool = False, ignore_label: int = 10):

    print(' ------- VALIDATING ------- ')
    pbar = tqdm(total=len(val_dataloader))
    pbar.set_description('validating: ')
    model.eval()

    loss_list = []
    total_true_positives = None
    total_true_negatives = None
    total_false_positives = None
    total_false_negatives = None

    with torch.no_grad():
        for val_iter, val_data in enumerate(val_dataloader):

            X_val, y_val = val_data
            X_val = X_val.float().to(device=device)
            y_val = y_val.long().to(device=device)      
            
            output = model(X_val)
            loss = criterion(output, y_val)
            predict_labels = torch.argmax(output, dim=1)

            if visualize:
                # visualize the first image form the batch
                vis_labels = predict_labels[0, :, :].cpu().clone()
                img_vis = X_val[0, :, :, :].cpu().clone()
                img_vis[:, vis_labels != 1] = 0
                imshow(img_vis)


            # compute statistics for metrics
            loss_list.append(loss.cpu().item())
            true_positives, true_negatives, false_positives, false_negatives = get_cls_stats(
                predict_labels=predict_labels.cpu(),
                labels=y_val.cpu(),
                unique_labels=unique_labels,
                ignore_label=ignore_label
            )
            if total_true_positives is None:
                total_true_positives = true_positives
                total_true_negatives = true_negatives
                total_false_positives = false_positives
                total_false_negatives = false_negatives
            else:
                total_true_positives += true_positives
                total_true_negatives += true_negatives
                total_false_positives += false_positives
                total_false_negatives += false_negatives
            pbar.update(1)
    pbar.close()
    model.train()

    return torch.tensor(loss_list), total_true_positives, total_true_negatives, total_false_positives, total_false_negatives

    

    