import cv2 as cv
import time
import sys
import collections
import os
from pathlib import Path

import motion_detector.image_processing as image_processing
import motion_detector.settings_detector as settings_detector  # Импортируем только для параметров обработки
import motion_detector.detector_IO as detector_IO

import motion_detector.detection_processing as detection_processing

def get_video_parameters(cap: cv.VideoCapture):
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    return fps, width, height, frame_count

def print_progress(frame_counter, total_frames, additional_info=""):
    percent = (frame_counter / total_frames) * 100
    sys.stdout.write(f"\rProgress: {percent:.1f}% | {additional_info}")
    sys.stdout.flush()

def apply_settings_for_night(bs: cv.BackgroundSubtractorMOG2, frame):
    #bs.setBackgroundRatio(settings_detector.NIGHT_BACKGROUND_RATIO)
    frame = image_processing.adaptiveHe(frame, 
                                      contrast=settings_detector.NIGHT_CONTRAST_FACTOR, 
                                      tile=settings_detector.NIGHT_TILE_GRID)
    mode = "night vision"
    return bs, frame, mode

def apply_settings_for_day(bs: cv.BackgroundSubtractorMOG2, frame):
    #bs.setVarThreshold(settings_detector.DAY_VAR_THRESHOLD)
    frame = image_processing.gammac(frame, settings_detector.DAY_GAMMA_TARGET)
    mode = 'day light'
    return bs, frame, mode

def set_background_parameters(bs: cv.BackgroundSubtractorMOG2):
    """Устанавливает параметры фонового вычитателя из settings.py"""
    bs.setBackgroundRatio(settings_detector.bs_background_ratio)
    bs.setComplexityReductionThreshold(settings_detector.bs_complexity_reduction_threshold)
    bs.setVarThresholdGen(settings_detector.bs_var_threshold_gen)
    bs.setNMixtures(settings_detector.bs_NMixtures)
    bs.setShadowThreshold(settings_detector.bs_shadow_threshold)
    bs.setShadowValue(settings_detector.bs_shadow_value)
    bs.setVarInit(settings_detector.bs_VarInit)
    bs.setVarMax(settings_detector.bs_VarMax)
    bs.setVarMin(settings_detector.bs_VarMin)
    return bs

def process_video(
    input_path: str,
    output_video_path: str,
    output_json_path: str = None,
    show_progress: bool = True,
    save_frames: bool = False,
    progress_callback=None 
) -> str:
    """
    Основная функция обработки видео
    
    Args:
        input_path: путь к входному видео
        output_video_path: путь для сохранения результата
        output_json_path: путь для сохранения JSON с детекциями (опционально)
        show_progress: показывать прогресс в консоли
        save_frames: сохранять ли отдельные кадры (для отладки)
    
    Returns:
        Путь к обработанному видео
    """
    # Проверка входных файлов
    print(f"Начало обработки: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Создаем директории для выходных файлов
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    # Инициализация видео
    cap = cv.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {input_path}")
    
    fps, width, height, total_frames = get_video_parameters(cap)
    if total_frames <= 0:
        raise ValueError("Could not determine video frame count")
    
    # Инициализация VideoWriter
    fourcc = cv.VideoWriter_fourcc(*'avc1')
    out = cv.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Инициализация фонового вычитателя
    #bs = cv.createBackgroundSubtractorMOG2(
       # history=settings_detector.bs_history,
      #  varThreshold=settings_detector.bs_var_threshold,
     #   detectShadows=settings_detector.bs_detect_shadows
    #)

    bs_YCrCb = image_processing.BackgroundSubtractor_YCbCr(settings_detector.bs_history, width, height)
    #bs = set_background_parameters(bs)
    bs = None
    
    # Основные переменные
    frame_counter = 0
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    start_time = time.perf_counter()
    detections = collections.deque(maxlen=settings_detector.buffer_size)
    night_mode = False
    
    if output_json_path:
        detector_IO.clear_output_file(output_json_path)
    
    if show_progress:
        print(f"\nProcessing video: {os.path.basename(input_path)}")
        print(f"Total frames: {total_frames}")
    
    # Главный цикл обработки
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_counter += 1
        current_time = time.perf_counter()
        processing_speed = 1 / (current_time - start_time)

        if progress_callback and frame_counter % 5 == 0:  # Обновляем каждые 5 кадров
            processing_speed = 1 / (time.perf_counter() - start_time) if frame_counter > 1 else 0
            progress_callback(frame_counter, total_frames, f"Speed: {processing_speed:.1f}fps")
        
        if show_progress:
            print_progress(
                frame_counter, 
                total_frames, 
                f"Frame: {frame_counter}/{total_frames} | Speed: {processing_speed:.1f}fps"
            )
        
        # Проверка ночного режима
        if frame_counter % settings_detector.check_night_per_frames == 0 or frame_counter == 1:
            night_mode = image_processing.nightVision(frame)
        
        # Обработка кадра
        processing_frame = frame.copy()
        if night_mode:
            bs, processing_frame, _ = apply_settings_for_night(bs, processing_frame)
            nf_threshold = settings_detector.nf_threshold_night
        else:
            bs, processing_frame, _ = apply_settings_for_day(bs, processing_frame)
            nf_threshold = settings_detector.nf_threshold_day
        
        # Вычитание фона и детекция
        #foreground = bs.apply(processing_frame)
        bs_YCrCb.add_frame(processing_frame)
        bs_YCrCb.create_background()
        foreground = bs_YCrCb.extract_foreground(processing_frame)
        
        frame_data = []
        if frame_counter > settings_detector.frames_skip:
            foreground = image_processing.foreground_postprocessing(foreground)
            contours, _ = cv.findContours(
                foreground,
                settings_detector.CONTOUR_RETRIEVAL_MODE,
                settings_detector.CONTOUR_APPROX_METHOD
            )
            frame_data = detection_processing.detect(frame_data, contours)
        
        # Обработка детекций
        outer_detections = detection_processing.remove_nested_detections(
            frame_data,
            settings_detector.NESTED_DETECTION_OVERLAP_THRESHOLD
        )
        detections.append(outer_detections)
        detections, _, _, _ = detection_processing.assignIDs(detections, nf_threshold)
        detections, _ = detection_processing.completeIDs(detections, nf_threshold)
        detections, _, _, _ = detection_processing.validateObjs(
            detections, frame_counter, fps, nf_threshold
        )
        
        # Сохранение результатов
        if output_json_path:
            detector_IO.print_frame_detection(
                output_json_path,
                detections[-1],
                frame_counter
            )
        
        # Визуализация и сохранение
        frame_with_boxes = detector_IO.show_boxes(frame.copy(), detections[-1], fps)
        out.write(frame_with_boxes)
        
        if save_frames:
            detector_IO.save_frame_with_boxes(
                frame_with_boxes,
                frame_counter,
                len(str(total_frames))
            )
    
    # Завершение
    print(f"Обработка завершена: {output_video_path}")
    if show_progress:
        print_progress(frame_counter, total_frames, "Completed!")
        print(f"\nAverage processing speed: {frame_counter/(time.perf_counter()-start_time):.1f} fps")
    
    cap.release()
    out.release()
    cv.destroyAllWindows()
    
    return output_video_path

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Moving objects detection')
    parser.add_argument('--input', required=True, help='Input video file path')
    parser.add_argument('--output-video', required=True, help='Output video file path')
    parser.add_argument('--output-json', help='Output JSON file path')
    parser.add_argument('--show-progress', action='store_true', help='Show progress in console')
    parser.add_argument('--save-frames', action='store_true', help='Save individual frames')
    
    args = parser.parse_args()
    
    try:
        result_path = process_video(
            input_path=args.input,
            output_video_path=args.output_video,
            output_json_path=args.output_json,
            show_progress=args.show_progress,
            save_frames=args.save_frames
        )
        print(f"\nProcessing complete. Result saved to: {result_path}")
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)