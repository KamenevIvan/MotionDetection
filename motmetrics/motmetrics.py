import numpy as np
from typing import Dict, Tuple


def load_mot_file(file_path: str) -> np.ndarray:
    return np.loadtxt(file_path, delimiter=',')


def calculate_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Вычисляет Intersection over Union (IoU) для двух bounding box.

    Args:
        bbox1 (np.ndarray): Первый bounding box в формате [x, y, width, height].
        bbox2 (np.ndarray): Второй bounding box в формате [x, y, width, height].

    Returns:
        float: Значение IoU.
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x_inter = max(x1, x2)
    y_inter = max(y1, y2)
    w_inter = min(x1 + w1, x2 + w2) - x_inter
    h_inter = min(y1 + h1, y2 + h2) - y_inter

    if w_inter <= 0 or h_inter <= 0:
        return 0.0

    area_inter = w_inter * h_inter
    area1 = w1 * h1
    area2 = w2 * h2
    area_union = area1 + area2 - area_inter

    return area_inter / area_union


def compute_iou_matrix(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """
    Вычисляет матрицу IoU между двумя наборами bounding box.

    Args:
        bboxes1 (np.ndarray): Первый набор bounding box.
        bboxes2 (np.ndarray): Второй набор bounding box.

    Returns:
        np.ndarray: Матрица IoU.
    """
    iou_matrix = np.zeros((len(bboxes1), len(bboxes2)))
    for i, bbox1 in enumerate(bboxes1):
        for j, bbox2 in enumerate(bboxes2):
            iou_matrix[i, j] = calculate_iou(bbox1, bbox2)
    return iou_matrix


def match_objects(iou_matrix: np.ndarray, iou_threshold: float = 0.5) -> Dict[int, int]:
    """
    Сопоставляет объекты на основе IoU.

    Args:
        iou_matrix (np.ndarray): Матрица IoU.
        iou_threshold (float): Порог IoU для сопоставления.

    Returns:
        Dict[int, int]: Словарь сопоставлений {gt_id: det_id}.
    """
    matches = {}
    for gt_id in range(iou_matrix.shape[0]):
        if iou_matrix.shape[1] > 0:
            best_det_id = np.argmax(iou_matrix[gt_id, :])
            if iou_matrix[gt_id, best_det_id] >= iou_threshold:
                matches[gt_id] = best_det_id
    return matches


def compute_localization_accuracy(gt: np.ndarray, det: np.ndarray) -> float:
    """
    Вычисляет локальную точность (LocA) на основе IoU.

    Args:
        gt (np.ndarray): Ground truth данные.
        det (np.ndarray): Данные детекций.

    Returns:
        float: Среднее значение IoU для всех сопоставлений.
    """
    ious = []
    for frame in np.unique(gt[:, 0]):
        gt_frame = gt[gt[:, 0] == frame]
        det_frame = det[det[:, 0] == frame]
        if len(gt_frame) == 0 or len(det_frame) == 0:
            continue

        iou_matrix = compute_iou_matrix(gt_frame[:, 2:6], det_frame[:, 2:6])
        matches = match_objects(iou_matrix, iou_threshold=0.5)

        for gt_id, det_id in matches.items():
            ious.append(iou_matrix[gt_id, det_id])

    return np.mean(ious) if ious else 0


def compute_association_metrics(gt: np.ndarray, det: np.ndarray) -> Tuple[float, float, float]:
    """
    Вычисляет метрики ассоциации (AssA, AssRe, AssPr).

    Args:
        gt (np.ndarray): Ground truth данные.
        det (np.ndarray): Данные детекций.

    Returns:
        Tuple[float, float, float]: AssA, AssRe, AssPr.
    """
    correct_associations = 0
    total_possible_associations = 0
    total_predicted_associations = 0 

    prev_matches = {}

    gt_object_frames = {obj_id: len(gt[gt[:, 1] == obj_id]) for obj_id in np.unique(gt[:, 1])}

    for frame in np.unique(gt[:, 0]):
        gt_frame = gt[gt[:, 0] == frame]
        det_frame = det[det[:, 0] == frame]

        if len(gt_frame) == 0 or len(det_frame) == 0:
            continue

        iou_matrix = compute_iou_matrix(gt_frame[:, 2:6], det_frame[:, 2:6])
        matches = match_objects(iou_matrix, iou_threshold=0.5)

        for obj_id in np.unique(gt_frame[:, 1]):
            if gt_object_frames[obj_id] > 1:
                total_possible_associations += 1

        for det_id in np.unique(det_frame[:, 1]):
            if len(det[det[:, 1] == det_id]) > 1:
                total_predicted_associations += 1

        for gt_id, det_id in matches.items():
            if gt_id in prev_matches and prev_matches[gt_id] == det_id:
                correct_associations += 1 
            prev_matches[gt_id] = det_id

    AssRe = correct_associations / total_possible_associations if total_possible_associations > 0 else 0
    AssPr = correct_associations / total_predicted_associations if total_predicted_associations > 0 else 0
    AssA = np.sqrt(AssRe * AssPr)

    return AssA, AssRe, AssPr


def compute_tracking_metrics(gt: np.ndarray, det: np.ndarray) -> Dict[str, float]:
    """
    Вычисляет метрики трекинга (MOTA, IDF1, MT, ML и др.).

    Args:
        gt (np.ndarray): Ground truth данные.
        det (np.ndarray): Данные детекций.

    Returns:
        Dict[str, float]: Словарь метрик.
    """
    TP = 0
    FP = 0
    FN = 0
    IDSW = 0
    Frag = 0
    prev_matches = {}

    total_gt = len(np.unique(gt[:, 1]))
    total_frames = len(np.unique(gt[:, 0]))

    gt_object_frames = {obj_id: len(gt[gt[:, 1] == obj_id]) for obj_id in np.unique(gt[:, 1])}

    tracked_frames = {obj_id: 0 for obj_id in np.unique(gt[:, 1])}

    for frame in np.unique(gt[:, 0]):
        gt_frame = gt[gt[:, 0] == frame]
        det_frame = det[det[:, 0] == frame]

        if len(gt_frame) == 0 or len(det_frame) == 0:
            FN += len(gt_frame)
            FP += len(det_frame)
            continue

        iou_matrix = compute_iou_matrix(gt_frame[:, 2:6], det_frame[:, 2:6])
        matches = match_objects(iou_matrix, iou_threshold=0.5)

        matched_gt_ids = set()

        for gt_id, det_id in matches.items():
            if gt_id in matched_gt_ids:
                continue

            if gt_id in prev_matches and prev_matches[gt_id] != det_id:
                IDSW += 1
                Frag += 1
            TP += 1
            prev_matches[gt_id] = det_id
            matched_gt_ids.add(gt_id)

            tracked_frames[gt_frame[gt_id, 1]] += 1

        FP += len(det_frame) - len(matches)
        FN += len(gt_frame) - len(matches)

    MT = 0
    ML = 0
    for obj_id in np.unique(gt[:, 1]):
        total_frames_obj = gt_object_frames[obj_id]
        tracked_frames_obj = tracked_frames.get(obj_id, 0)

        if tracked_frames_obj / total_frames_obj >= 0.8:
            MT += 1
        elif tracked_frames_obj / total_frames_obj <= 0.2:
            ML += 1

    MOTA = 1 - (FN + FP + IDSW) / total_gt if total_gt > 0 else 0
    IDF1 = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    Rcll = TP / (TP + FN) if (TP + FN) > 0 else 0
    Prcn = TP / (TP + FP) if (TP + FP) > 0 else 0
    FAF = FP / total_frames if total_frames > 0 else 0
    Hz = total_frames / (gt[-1, 0] - gt[0, 0] + 1) if (gt[-1, 0] - gt[0, 0] + 1) > 0 else 0

    LocA = compute_localization_accuracy(gt, det)

    AssA, AssRe, AssPr = compute_association_metrics(gt, det)

    return {
        "MOTA": MOTA,
        "IDF1": IDF1,
        "HOTA": AssA,
        "MT": MT,
        "ML": ML,
        "FP": FP,
        "FN": FN,
        "Rcll": Rcll,
        "Prcn": Prcn,
        "AssA": AssA,
        "AssRe": AssRe,
        "AssPr": AssPr,
        "LocA": LocA,
        "FAF": FAF,
        "ID Sw.": IDSW,
        "Frag": Frag,
        "Hz": Hz,
    }


def main():
    gt = load_mot_file('ground-truth.txt')
    det = load_mot_file('../results/mot-results.txt')

    metrics = compute_tracking_metrics(gt, det)

    print("\t".join(metrics.keys()))
    print("\t".join(f"{value:.1f}" for value in metrics.values()))


if __name__ == "__main__":
    main()