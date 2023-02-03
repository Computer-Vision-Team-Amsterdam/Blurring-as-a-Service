import numpy as np
import torch


def evaluation_per_size(tp_D, conf_D, pred_cls_D, target_cls_D, bounding_boxes, size):

    # absolute
    bounding_boxes = torch.cat(bounding_boxes, 0).cpu().numpy()  # to numpy
    areas = bounding_boxes[:, 2] * bounding_boxes[:, 3]
    # cast to relative
    # divide by (4000 x 4000)
    areas = areas / 16000000

    """
    # indices where pred class is 0
    # 0: person, 1: licence_plate
    idx_pred_cls_0 = np.where(target_cls_D == 0)
    idx_pred_cls_1 = np.where(target_cls_D == 1)

    # get areas_persons, areas_licences

    areas_cls_0 = areas[idx_pred_cls_0]
    areas_cls_1 = areas[idx_pred_cls_1]

    plt.hist(areas_cls_1, bins=20, color='blue', alpha=0.5)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of bbox area')
    plt.savefig("../bbox_area_licence_plate.jpg")
    plt.show()
    """

    INTERVALS_CLS_0 = {
        "small": [0, 0.05],
        "medium": [0.05, 0.25],
        "large": [0.25, 1],
    }  # persons
    INTERVALS_CLS_1 = {
        "small": [0, 0.1],
        "medium": [0.1, 0.2],
        "large": [0.2, 0.5],
    }  # licence_plates

    # indices where prediction class is 0 or 1
    idx_pred_cls_0 = np.where(pred_cls_D == 0)
    idx_pred_cls_1 = np.where(pred_cls_D == 1)

    # indices of predicted areas where class 0 is a specific size (either small, medium or large)
    idx_small_area_cls_0 = np.where(
        (areas >= INTERVALS_CLS_0[size][0]) & (areas < INTERVALS_CLS_0[size][1])
    )[0]
    # indices of predicted areas where class 1 is a specific size (either small, medium or large)
    idx_small_area_cls_1 = np.where(
        (areas >= INTERVALS_CLS_1[size][0]) & (areas < INTERVALS_CLS_1[size][1])
    )[0]

    # indices where predicted area is a specific size AND class is 0
    idx_pred_cls_0_small = np.intersect1d(idx_pred_cls_0, idx_small_area_cls_0)
    # indices where predicted area is a specific size AND class is 1
    idx_pred_cls_1_small = np.intersect1d(idx_pred_cls_1, idx_small_area_cls_1)

    # indices where predicted area is a specific size
    idx_pred_small = np.concatenate((idx_pred_cls_0_small, idx_pred_cls_1_small))

    # create subset with correct indices
    tp_D = tp_D  # I think here we need all indices where TRUE area is a specific size.
    conf_D = conf_D[idx_pred_small]
    pred_cls_D = pred_cls_D[idx_pred_small]
    target_cls_D = target_cls_D  # I think here we need all indices where TRUE area is a specific size

    stats = [tp_D, conf_D, pred_cls_D, target_cls_D]

    return stats
