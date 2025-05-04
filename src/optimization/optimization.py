import optuna
import cv2 as cv
import numpy as np
import json
import time
import os
from functools import partial

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import image_processing
import detection_processing
import settings
from annotation_utils import AnnotationLoader, DetectionEvaluator

class AnnotationBasedEvaluator:
    """Класс для оценки детектора с использованием аннотаций"""
    
    def __init__(self, video_path, annotation_path, iou_threshold=0.5):
        """
        Инициализация оценщика детектора
        
        Args:
            video_path (str): Путь к видеофайлу
            annotation_path (str): Путь к файлу с аннотациями
            iou_threshold (float): Порог IoU для считания детекции правильной
        """
        self.video_path = video_path
        self.cap = cv.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
        
        self.fps, self.width, self.height = self._get_video_parameters()
        self.annotation_loader = AnnotationLoader(annotation_path)
        self.detection_evaluator = DetectionEvaluator(iou_threshold=iou_threshold)
        
        self.annotated_frames = self.annotation_loader.get_frame_ids()
        print(f"Найдено {len(self.annotated_frames)} кадров с аннотациями")
    
    def _get_video_parameters(self):
        """
        Получение параметров видео
        
        Returns:
            tuple: (fps, width, height)
        """
        fps = self.cap.get(cv.CAP_PROP_FPS)
        width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        return fps, width, height
    
    def _create_detector(self, params):
        """
        Создание детектора с заданными параметрами
        
        Args:
            params (dict): Параметры детектора
            
        Returns:
            object: Объект детектора
        """
        if params['color_space'] == 'HSV':
            detector = image_processing.BackgroundSubtractor_HSV(
                buffer_size=params['buffer_size'],
                frame_width=self.width,
                frame_height=self.height
            )
            detector.threshold_h = params['threshold_h']
            detector.threshold_s = params['threshold_s']
            detector.threshold_v = params['threshold_v']
            detector.blur = params['blur']
            detector.minimum_are_contours = params['min_area']
        elif params['color_space'] == 'YCbCr':
            detector = image_processing.BackgroundSubtractor_YCbCr(
                buffer_size=params['buffer_size'],
                frame_width=self.width,
                frame_height=self.height
            )
            detector.threshold_y = params['threshold_y']
            detector.threshold_chroma = params['threshold_chroma']
            detector.blur = params['blur']
            detector.min_contour_area = params['min_area']
        else:  # GRAY
            detector = image_processing.BackgroundSubtractor(
                buffer_size=params['buffer_size'],
                frame_width=self.width,
                frame_height=self.height
            )
            detector.threshold_value = params['threshold']
            detector.blur = params['blur']
            detector.minimum_are_contours = params['min_area']
        
        return detector
    
    def _detection_to_dict(self, detection):
        """
        Преобразование объекта Detection в словарь
        
        Args:
            detection (Detection): Объект Detection
            
        Returns:
            dict: Словарь с параметрами детекции
        """
        return {
            'x': detection.x,
            'y': detection.y,
            'width': detection.width,
            'height': detection.height
        }
    
    def evaluate_detector(self, params, max_frames=None, visualize=False, output_dir=None):
        """
        Оценка детектора с заданными параметрами
        
        Args:
            params (dict): Параметры детектора
            max_frames (int, optional): Максимальное количество кадров для оценки
            visualize (bool, optional): Флаг для визуализации результатов
            output_dir (str, optional): Директория для сохранения визуализаций
            
        Returns:
            dict: Словарь с метриками оценки
        """
        detector = self._create_detector(params)

        merge_distance_threshold = params.get('merge_distance_threshold', settings.merge_distance_threshold)
        overlap_threshold = params.get('overlap_threshold', settings.overlap_threshold)
        
        total_metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
        processing_time = 0.0
        frame_count = 0
        
        if visualize and output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        
        buffer_frames = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_id = int(self.cap.get(cv.CAP_PROP_POS_FRAMES))
            
            detector.add_frame(frame)
            buffer_frames += 1
            
            if buffer_frames <= params['buffer_size']:
                continue
            
            if max_frames is not None and frame_count >= max_frames:
                break
            
            if frame_id not in self.annotated_frames:
                continue
            
            start_time = time.time()
            
            detector.create_background()
            foreground = detector.extract_foreground(frame)
            
            contours, _ = cv.findContours(foreground, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            framedata = []
            detection_processing.detect(framedata, contours)
            merged_detections = detection_processing.merge_close_detections(framedata, merge_distance_threshold)
            final_detections = detection_processing.remove_nested_detections(merged_detections, overlap_threshold)
            detections_dict = [self._detection_to_dict(det) for det in final_detections]
            
            ground_truth = self.annotation_loader.get_annotations_for_frame(frame_id)
            
            metrics = self.detection_evaluator.evaluate_detections(detections_dict, ground_truth)
            
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            
            processing_time += time.time() - start_time
            frame_count += 1
            
            if visualize and output_dir:
                eval_frame = self.detection_evaluator.visualize_evaluation(frame, detections_dict, ground_truth)
                
                text = f"Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}, F1: {metrics['f1_score']:.2f}"
                cv.putText(eval_frame, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                output_path = os.path.join(output_dir, f"frame_{frame_id:06d}.jpg")
                cv.imwrite(output_path, eval_frame)
        
        self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        
        if frame_count == 0:
            return {
                'score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'processing_time': 0.0
            }
        
        avg_metrics = {key: value / frame_count for key, value in total_metrics.items()}
        avg_processing_time = processing_time / frame_count
        
        score = (avg_metrics['f1_score'] * 0.7 + 
                 avg_metrics['precision'] * 0.15 + 
                 avg_metrics['recall'] * 0.15) * (1.0 / (1.0 + avg_processing_time * 0.1))
        
        return {
            'score': score,
            'precision': avg_metrics['precision'],
            'recall': avg_metrics['recall'],
            'f1_score': avg_metrics['f1_score'],
            'true_positives': avg_metrics['true_positives'],
            'false_positives': avg_metrics['false_positives'],
            'false_negatives': avg_metrics['false_negatives'],
            'processing_time': avg_processing_time
        }

def objective(trial, evaluator, max_frames=None):
    """
    Целевая функция для Optuna
    
    Args:
        trial (optuna.Trial): Объект Trial из Optuna
        evaluator (AnnotationBasedEvaluator): Оценщик детектора
        max_frames (int, optional): Максимальное количество кадров для оценки
        
    Returns:
        float: Оценка детектора
    """
    params = {
        'color_space': trial.suggest_categorical('color_space', ['HSV', 'YCbCr', 'GRAY']),
        'buffer_size': trial.suggest_int('buffer_size', 10, 100),
        'blur': trial.suggest_int('blur', 3, 15, step=2),
        'min_area': trial.suggest_int('min_area', 10, 200),
        
        'merge_distance_threshold': trial.suggest_int('merge_distance_threshold', 1, 20),
        'overlap_threshold': trial.suggest_float('overlap_threshold', 0.3, 0.9)
    }
    
    if params['color_space'] == 'HSV':
        params.update({
            'threshold_h': trial.suggest_int('threshold_h', 50, 200),
            'threshold_s': trial.suggest_int('threshold_s', 10, 100),
            'threshold_v': trial.suggest_int('threshold_v', 50, 200)
        })
    elif params['color_space'] == 'YCbCr':
        params.update({
            'threshold_y': trial.suggest_int('threshold_y', 10, 100),
            'threshold_chroma': trial.suggest_int('threshold_chroma', 5, 50)
        })
    else:  # GRAY
        params.update({
            'threshold': trial.suggest_int('threshold', 10, 100)
        })
    
    results = evaluator.evaluate_detector(params, max_frames=max_frames)
    print(f"Trial {trial.number}: F1={results['f1_score']:.4f}, P={results['precision']:.4f}, R={results['recall']:.4f}, Time={results['processing_time']:.4f}s")
    
    return results['score']

def optimize_parameters(video_path, annotation_path, n_trials=100, max_frames=None, output_dir=None):
    """
    Оптимизация параметров детектора с использованием аннотаций
    
    Args:
        video_path (str): Путь к видеофайлу
        annotation_path (str): Путь к файлу с аннотациями
        n_trials (int, optional): Количество испытаний для оптимизации
        max_frames (int, optional): Максимальное количество кадров для оценки
        output_dir (str, optional): Директория для сохранения результатов
        
    Returns:
        dict: Лучшие параметры детектора
    """
    print(f"Начинаем оптимизацию параметров детектора...")
    print(f"Видео: {video_path}")
    print(f"Аннотации: {annotation_path}")
    print(f"Количество испытаний: {n_trials}")
    
    evaluator = AnnotationBasedEvaluator(video_path, annotation_path)
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    study = optuna.create_study(direction='maximize')
    
    study.optimize(
        partial(objective, evaluator=evaluator, max_frames=max_frames),
        n_trials=n_trials
    )
    
    best_params = study.best_params
    best_value = study.best_value
    
    print("\nОптимизация завершена!")
    print(f"Лучшие параметры: {best_params}")
    print(f"Лучшая оценка: {best_value:.4f}")
    
    print("\nОценка лучших параметров на всем видео...")
    best_metrics = evaluator.evaluate_detector(
        best_params,
        visualize=True,
        output_dir=os.path.join(output_dir, 'best_visualization') if output_dir else None
    )
    
    print(f"F1 Score: {best_metrics['f1_score']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"True Positives: {best_metrics['true_positives']:.2f}")
    print(f"False Positives: {best_metrics['false_positives']:.2f}")
    print(f"False Negatives: {best_metrics['false_negatives']:.2f}")
    print(f"Processing Time: {best_metrics['processing_time']:.4f} seconds per frame")
    
    if output_dir:
        params_path = os.path.join(output_dir, 'best_params.json')
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        history_path = os.path.join(output_dir, 'optimization_history.csv')
        df = study.trials_dataframe()
        df.to_csv(history_path, index=False)
        
        print(f"\nЛучшие параметры сохранены в {params_path}")
        print(f"История оптимизации сохранена в {history_path}")
    
    return best_params