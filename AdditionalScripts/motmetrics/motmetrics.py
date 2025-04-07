import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, Tuple, Set, List, Any, Optional
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FormatError(Exception):
    """Custom exception for unsupported MOT format types."""
    pass


class ResultsWriter:
    """Handle writing metrics results to CSV files."""
    
    @staticmethod
    def write_to_csv(metrics: Dict[str, float], output_path: str) -> None:
        """Write metrics dictionary to CSV file.
        
        Args:
            metrics: Dictionary of metric names and values
            output_path: Path to output CSV file
        """
        try:
            file_exists = Path(output_path).exists()
            
            with open(output_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                
                # Write header if file doesn't exist
                if not file_exists:
                    writer.writerow(['timestamp'] + list(metrics.keys()))
                
                # Write data row
                writer.writerow([datetime.now().isoformat()] + list(metrics.values()))
                
            logger.info(f"Results saved to: {output_path}")
        except IOError as e:
            logger.error(f"Failed to write results to {output_path}: {str(e)}")
            raise


class MOTMetricsCalculator:
    """Calculate multiple object tracking metrics from MOT challenge files."""

    def __init__(self):
        self.gt_data = None
        self.det_data = None

    def load_mot_file(self, file_path: str, format_type: str = 'auto') -> np.ndarray:
        """Load MOT format file with automatic format detection."""
        try:
            data = np.loadtxt(file_path, delimiter=',')
        except IOError as e:
            raise IOError(f"Could not load MOT file: {file_path}") from e

        if format_type == 'auto':
            if data.shape[1] >= 6 and np.all(data[:, 5] == 1):
                format_type = 'short' if data.shape[1] == 6 else 'extended_short'
            elif data.shape[1] >= 7:
                format_type = 'extended' if np.all(data[:, 6:] == -1) else 'standard'
            else:
                format_type = 'standard'

        if format_type in ('short', 'extended_short'):
            extended_data = np.zeros((data.shape[0], 7 + max(0, data.shape[1]-6)))
            extended_data[:, 0] = data[:, 0]  # frame
            extended_data[:, 1] = 1 if format_type == 'short' else data[:, 1]  # objID
            extended_data[:, 2:6] = data[:, 1:5]  # x,y,w,h
            extended_data[:, 6] = 1 - data[:, 5] if format_type == 'short' else data[:, 5]  # conf
            if data.shape[1] > 6:
                extended_data[:, 7:] = data[:, 6:]
            return extended_data

        elif format_type in ('extended', 'standard'):
            return data[:, :7] if format_type == 'standard' else data

        raise FormatError(f"Unknown format type: {format_type}")

    @staticmethod
    def calculate_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate Intersection over Union for two bounding boxes."""
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

    @staticmethod
    def compute_iou_matrix(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
        """Compute IoU matrix between two sets of bounding boxes."""
        iou_matrix = np.zeros((len(bboxes1), len(bboxes2)))
        for i, bbox1 in enumerate(bboxes1):
            for j, bbox2 in enumerate(bboxes2):
                iou_matrix[i, j] = MOTMetricsCalculator.calculate_iou(bbox1, bbox2)
        return iou_matrix

    @staticmethod
    def match_objects(iou_matrix: np.ndarray, iou_threshold: float = 0.5) -> Dict[int, int]:
        """Match objects between ground truth and detections using IoU matrix."""
        matches = {}
        for gt_id in range(iou_matrix.shape[0]):
            if iou_matrix.shape[1] > 0:
                best_det_id = np.argmax(iou_matrix[gt_id, :])
                if iou_matrix[gt_id, best_det_id] >= iou_threshold:
                    matches[gt_id] = best_det_id
        return matches

    def validate_data(self, gt: np.ndarray, det: np.ndarray) -> None:
        """Validate input data and log statistics."""
        logger.info("\nData statistics:")
        logger.info(f"GT objects: {len(np.unique(gt[:,1]))} across {len(np.unique(gt[:,0]))} frames")
        logger.info(f"Det objects: {len(np.unique(det[:,1]))} across {len(np.unique(det[:,0]))} frames")
        
        only_in_gt = set(np.unique(gt[:,0])) - set(np.unique(det[:,0]))
        only_in_det = set(np.unique(det[:,0])) - set(np.unique(gt[:,0]))
        
        if only_in_gt:
            logger.warning(f"Frames only in GT: {sorted(only_in_gt)}")
        if only_in_det:
            logger.warning(f"Frames only in Det: {sorted(only_in_det)}")

    def compute_tracking_metrics(self, gt: np.ndarray, det: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive set of tracking metrics."""
        TP = 0
        FP = 0
        FN = 0
        IDSW = 0
        Frag = 0
        prev_matches = {}
        id_mapping = {}
        fp_frames = set()
        id_switch_details = []

        total_gt = len(np.unique(gt[:, 1]))
        all_frames = set(np.unique(gt[:, 0])).union(set(np.unique(det[:, 0])))
        total_frames = len(all_frames)

        gt_object_frames = {obj_id: len(gt[gt[:, 1] == obj_id]) for obj_id in np.unique(gt[:, 1])}
        tracked_frames = {obj_id: 0 for obj_id in np.unique(gt[:, 1])}
        LocA_list = []

        for frame in sorted(all_frames):
            gt_frame = gt[gt[:, 0] == frame]
            det_frame = det[det[:, 0] == frame]

            if len(gt_frame) == 0:
                if frame not in np.unique(gt[:, 0]):
                    FP += len(det_frame)
                    if len(det_frame) > 0:
                        fp_frames.add(frame)
                continue
                
            iou_matrix = self.compute_iou_matrix(gt_frame[:, 2:6], det_frame[:, 2:6])
            matches = self.match_objects(iou_matrix, iou_threshold=0.5)

            current_mapping = {}
            for gt_idx, det_idx in matches.items():
                gt_obj_id = gt_frame[gt_idx, 1]
                det_obj_id = det_frame[det_idx, 1]
                current_mapping[det_obj_id] = gt_obj_id

            current_fp = 0
            for i in range(len(det_frame)):
                det_obj_id = det_frame[i, 1]
                if i not in matches.values():
                    if det_obj_id not in id_mapping:
                        current_fp += 1
                        fp_frames.add(frame)
            
            FP += current_fp

            for gt_idx, det_idx in matches.items():
                gt_obj_id = gt_frame[gt_idx, 1]
                det_obj_id = det_frame[det_idx, 1]
                
                if gt_obj_id in prev_matches:
                    prev_det_id = prev_matches[gt_obj_id]
                    if prev_det_id != det_obj_id:
                        if det_obj_id not in id_mapping or id_mapping[det_obj_id] != gt_obj_id:
                            IDSW += 1
                            Frag += 1
                            id_switch_details.append(
                                f"Frame {frame}: GT {gt_obj_id} switched from {prev_det_id} to {det_obj_id}"
                            )
                
                TP += 1
                prev_matches[gt_obj_id] = det_obj_id
                id_mapping[det_obj_id] = gt_obj_id
                tracked_frames[gt_obj_id] += 1
                LocA_list.append(iou_matrix[gt_idx, det_idx])

            FN += len(gt_frame) - len(matches)

        LocA = np.mean(LocA_list) if LocA_list else 0
        AssA, AssRe, AssPr = self.compute_association_metrics(gt, det)
        HOTA = np.sqrt(AssA * LocA) if (AssA > 0 and LocA > 0) else 0

        MT = sum(1 for obj_id in gt_object_frames 
                if tracked_frames.get(obj_id, 0)/gt_object_frames[obj_id] >= 0.5)
        ML = sum(1 for obj_id in gt_object_frames 
                if tracked_frames.get(obj_id, 0)/gt_object_frames[obj_id] <= 0.2)

        MOTA = max(0, 1 - (FN + FP + IDSW)/max(1, total_gt))
        IDF1 = (2*TP)/(2*TP + FP + FN) if (2*TP + FP + FN) > 0 else 0
        Rcll = TP/(TP + FN) if (TP + FN) > 0 else 0
        Prcn = TP/(TP + FP) if (TP + FP) > 0 else 0
        FAF = FP/total_frames if total_frames > 0 else 0

        logger.debug("\nError details:")
        logger.debug(f"FP frames: {sorted(fp_frames)}")
        logger.debug("ID switches:")
        for switch in id_switch_details:
            logger.debug(f"  {switch}")

        return {
            "MOTA": MOTA,
            "IDF1": IDF1,
            "HOTA": HOTA,
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
        }
        
    def compute_association_metrics(self, gt: np.ndarray, det: np.ndarray) -> Tuple[float, float, float]:
        """Compute association metrics (AssA, AssRe, AssPr)."""
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

            iou_matrix = self.compute_iou_matrix(gt_frame[:, 2:6], det_frame[:, 2:6])
            matches = self.match_objects(iou_matrix, iou_threshold=0.5)

            for obj_id in np.unique(gt_frame[:, 1]):
                if gt_object_frames[obj_id] > 1:
                    total_possible_associations += 1

            for det_id in np.unique(det_frame[:, 1]):
                if len(det[det[:, 1] == det_id]) > 1:
                    total_predicted_associations += 1

            for gt_idx, det_idx in matches.items():
                gt_obj_id = gt_frame[gt_idx, 1]
                det_obj_id = det_frame[det_idx, 1]
                if gt_obj_id in prev_matches and prev_matches[gt_obj_id] == det_obj_id:
                    correct_associations += 1
                prev_matches[gt_obj_id] = det_obj_id

        AssRe = correct_associations / total_possible_associations if total_possible_associations > 0 else 0
        AssPr = correct_associations / total_predicted_associations if total_predicted_associations > 0 else 0
        AssA = np.sqrt(AssRe * AssPr)

        return AssA, AssRe, AssPr


def main():
    """Command line interface for MOT metrics calculation."""
    parser = argparse.ArgumentParser(
        description='Calculate MOT tracking metrics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-g', '--ground-truth',
        required=True,
        help='Path to ground truth file'
    )
    parser.add_argument(
        '-d', '--detections',
        required=True,
        help='Path to detections file'
    )
    parser.add_argument(
        '-o', '--output',
        default='tracking_metrics.csv',
        help='Path to output CSV file'
    )
    parser.add_argument(
        '--gt-format',
        default='auto',
        choices=['auto', 'standard', 'short', 'extended'],
        help='Ground truth file format'
    )
    parser.add_argument(
        '--det-format',
        default='auto',
        choices=['auto', 'standard', 'short', 'extended'],
        help='Detections file format'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        calculator = MOTMetricsCalculator()
        
        logger.info("Loading ground truth data...")
        gt = calculator.load_mot_file(args.ground_truth, args.gt_format)
        
        logger.info("Loading detection data...")
        det = calculator.load_mot_file(args.detections, args.det_format)

        calculator.validate_data(gt, det)

        logger.info("Computing metrics...")
        metrics = calculator.compute_tracking_metrics(gt, det)

        logger.info("\nTracking Metrics:")
        for name, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{name:>8}: {value:.3f}")
            else:
                logger.info(f"{name:>8}: {value}")

        # Save results to CSV
        ResultsWriter.write_to_csv(metrics, args.output)

    except Exception as e:
        logger.critical(f"Error calculating metrics: {str(e)}", exc_info=True)
        exit(1)


if __name__ == '__main__':
    main()
