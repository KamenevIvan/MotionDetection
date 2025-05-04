import numpy as np
import cv2 as cv
import os

class AnnotationLoader:
    """Класс для загрузки и обработки аннотаций"""
    
    def __init__(self, annotation_path):
        """
        Инициализация загрузчика аннотаций
        
        Args:
            annotation_path (str): Путь к файлу с аннотациями
        """
        self.annotation_path = annotation_path
        self.annotations = self._load_annotations()
        
    def _load_annotations(self):
        """
        Загрузка аннотаций из файла
        
        Формат аннотаций: [frameID, xmin, ymin, width, height, isLost]
        
        Returns:
            dict: Словарь аннотаций, где ключ - номер кадра, значение - список аннотаций
        """
        annotations = {}
        
        if not os.path.exists(self.annotation_path):
            print(f"Файл аннотаций не найден: {self.annotation_path}")
            return annotations
        
        try:
            with open(self.annotation_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 6:
                        frame_id = int(parts[0])
                        xmin = float(parts[1])
                        ymin = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        is_lost = int(parts[5])
                        
                        if is_lost == 0:
                            if frame_id not in annotations:
                                annotations[frame_id] = []
                            
                            annotations[frame_id].append({
                                'x': xmin,
                                'y': ymin,
                                'width': width,
                                'height': height
                            })
            
            print(f"Загружены аннотации для {len(annotations)} кадров")
        except Exception as e:
            print(f"Ошибка при загрузке аннотаций: {e}")
        
        return annotations
    
    def get_annotations_for_frame(self, frame_id):
        """
        Получение аннотаций для конкретного кадра
        
        Args:
            frame_id (int): Номер кадра
            
        Returns:
            list: Список аннотаций для кадра или пустой список, если аннотаций нет
        """
        return self.annotations.get(frame_id, [])
    
    def get_frame_ids(self):
        """
        Получение списка номеров кадров, для которых есть аннотации
        
        Returns:
            list: Список номеров кадров
        """
        return sorted(list(self.annotations.keys()))
    
    def visualize_annotations(self, frame, frame_id, color=(0, 255, 0), thickness=2):
        """
        Визуализация аннотаций на кадре
        
        Args:
            frame (numpy.ndarray): Кадр для визуализации
            frame_id (int): Номер кадра
            color (tuple): Цвет рамки (B, G, R)
            thickness (int): Толщина линии
            
        Returns:
            numpy.ndarray: Кадр с нарисованными аннотациями
        """
        annotations = self.get_annotations_for_frame(frame_id)
        frame_with_annotations = frame.copy()
        
        for annotation in annotations:
            x = int(annotation['x'])
            y = int(annotation['y'])
            w = int(annotation['width'])
            h = int(annotation['height'])
            
            cv.rectangle(frame_with_annotations, (x, y), (x + w, y + h), color, thickness)
            
        return frame_with_annotations


class DetectionEvaluator:
    """Класс для оценки качества детекций с использованием аннотаций"""
    
    def __init__(self, iou_threshold=0.5):
        """
        Инициализация оценщика детекций
        
        Args:
            iou_threshold (float): Порог IoU для считания детекции правильной
        """
        self.iou_threshold = iou_threshold
    
    def calculate_iou(self, box1, box2):
        """
        Вычисление IoU (Intersection over Union) между двумя боксами
        
        Args:
            box1 (dict): Первый бокс с ключами 'x', 'y', 'width', 'height'
            box2 (dict): Второй бокс с ключами 'x', 'y', 'width', 'height'
            
        Returns:
            float: Значение IoU в диапазоне [0, 1]
        """
        # Координаты углов первого бокса
        x1_min = box1['x']
        y1_min = box1['y']
        x1_max = x1_min + box1['width']
        y1_max = y1_min + box1['height']
        
        # Координаты углов второго бокса
        x2_min = box2['x']
        y2_min = box2['y']
        x2_max = x2_min + box2['width']
        y2_max = y2_min + box2['height']
        
        x_min_intersect = max(x1_min, x2_min)
        y_min_intersect = max(y1_min, y2_min)
        x_max_intersect = min(x1_max, x2_max)
        y_max_intersect = min(y1_max, y2_max)
        
        if x_max_intersect < x_min_intersect or y_max_intersect < y_min_intersect:
            return 0.0
        
        intersection_area = (x_max_intersect - x_min_intersect) * (y_max_intersect - y_min_intersect)
        
        box1_area = box1['width'] * box1['height']
        box2_area = box2['width'] * box2['height']
        
        union_area = box1_area + box2_area - intersection_area
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou
    
    def evaluate_detections(self, detections, ground_truth):
        """
        Оценка детекций с использованием ground truth аннотаций
        
        Args:
            detections (list): Список детекций, каждая детекция - словарь с ключами 'x', 'y', 'width', 'height'
            ground_truth (list): Список ground truth аннотаций в том же формате
            
        Returns:
            dict: Словарь с метриками оценки (precision, recall, f1_score)
        """
        if not ground_truth:
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'true_positives': 0, 'false_positives': len(detections), 'false_negatives': 0}
        
        if not detections:
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'true_positives': 0, 'false_positives': 0, 'false_negatives': len(ground_truth)}
        
        gt_matched = [False] * len(ground_truth)
        det_matched = [False] * len(detections)
        
        iou_matrix = np.zeros((len(detections), len(ground_truth)))
        
        for i, detection in enumerate(detections):
            for j, gt in enumerate(ground_truth):
                iou_matrix[i, j] = self.calculate_iou(detection, gt)
        
        while True:
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_iou = iou_matrix[i, j]
            
            if max_iou < self.iou_threshold:
                break

            det_matched[i] = True
            gt_matched[j] = True

            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0
        
        true_positives = sum(det_matched)
        false_positives = len(detections) - true_positives
        false_negatives = len(ground_truth) - true_positives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def visualize_evaluation(self, frame, detections, ground_truth):
        """
        Визуализация результатов оценки на кадре
        
        Args:
            frame (numpy.ndarray): Кадр для визуализации
            detections (list): Список детекций
            ground_truth (list): Список ground truth аннотаций
            
        Returns:
            numpy.ndarray: Кадр с визуализацией
        """
        frame_with_eval = frame.copy()
        
        for gt in ground_truth:
            x = int(gt['x'])
            y = int(gt['y'])
            w = int(gt['width'])
            h = int(gt['height'])
            cv.rectangle(frame_with_eval, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        for det in detections:
            x = int(det['x'])
            y = int(det['y'])
            w = int(det['width'])
            h = int(det['height'])
            cv.rectangle(frame_with_eval, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        for det in detections:
            for gt in ground_truth:
                iou = self.calculate_iou(det, gt)
                if iou >= self.iou_threshold:
                    x = int(det['x'])
                    y = int(det['y'])
                    w = int(det['width'])
                    h = int(det['height'])
                    cv.rectangle(frame_with_eval, (x, y), (x + w, y + h), (255, 0, 0), 1)
                    break
        
        return frame_with_eval