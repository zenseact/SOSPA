import numpy as np


def calculate_average_precision(tpfp_score_list: dict, thresholds: list, num_gts: int):
    """Calculate average precision for a given label and return the result dictionary.

    Args:
        tpfp_score_list (dict): List of tp, fp, and score for each detection.
        thresholds (list): List of thresholds.
        num_gts (int): Number of ground truths for the label.

    Returns:
        tuple: Mean average precision (mAP) and a dictionary containing AP for each threshold.
    """
    sum_AP = 0
    result_dict = {}

    for thr in thresholds:
        tp_fp_score = [i[thr] for i in tpfp_score_list]
        tp_fp_score = np.vstack(tp_fp_score)  # (num_dets, 3)
        sort_inds = np.argsort(-tp_fp_score[:, -1])

        tp = tp_fp_score[sort_inds, 0]  # (num_dets,)
        fp = tp_fp_score[sort_inds, 1]  # (num_dets,)
        tp = np.cumsum(tp, axis=0)
        fp = np.cumsum(fp, axis=0)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts, eps)
        precisions = tp / np.maximum((tp + fp), eps)

        AP = average_precision(recalls, precisions, "area")

        sum_AP += AP
        result_dict[f"AP@{thr}"] = AP

    AP = sum_AP / len(thresholds)
    result_dict["AP"] = AP

    return AP, result_dict


def average_precision(recalls, precisions, mode="area"):
    """Calculate average precision.

    Args:
        recalls (ndarray): shape (num_dets, )
        precisions (ndarray): shape (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float: calculated average precision
    """

    recalls = recalls[np.newaxis, :]
    precisions = precisions[np.newaxis, :]

    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = 0.0

    if mode == "area":
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])

        ind = np.where(mrec[0, 1:] != mrec[0, :-1])[0]
        ap = np.sum((mrec[0, ind + 1] - mrec[0, ind]) * mpre[0, ind + 1])

    elif mode == "11points":
        for thr in np.arange(0, 1 + 1e-3, 0.1):
            precs = precisions[0, recalls[i, :] >= thr]
            prec = precs.max() if precs.size > 0 else 0
            ap += prec
        ap /= 11
    else:
        raise ValueError('Unrecognized mode, only "area" and "11points" are supported')

    return ap


def _check_recalls(recalls, tp, num_gts):
    for i, rec in enumerate(recalls):
        if rec > 1.1:
            raise ValueError(
                f"Recall value {rec} is greater than 1.0, which is invalid. Current tp: {tp[i]}, num_gts: {num_gts}"
            )


def _check_precision_recall_validity(precisions, recalls, tp, fp, num_gts):
    for prec in precisions:
        if prec > 1.0:
            raise ValueError(
                f"Precision value {prec} is greater than 1.0, which is invalid. Current tp: {np.sum(tp)}, fp: {np.sum(fp)}, num_gts: {num_gts}"
            )
    for rec in recalls:
        if rec > 1.1:
            raise ValueError(
                f"Recall value {rec} is greater than 1.0, which is invalid. Current tp: {np.sum(tp)}, fp: {np.sum(fp)}, num_gts: {num_gts}"
            )