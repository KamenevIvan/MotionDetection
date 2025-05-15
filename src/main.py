import cv2 as cv
import math
import time
import sys
import base64
import json
import collections
import argparse
import os

import settings
import detector_IO
import image_processing
import detection_processing

######## moving objects detector module #################
#   Version 8 from 29.11.2024 detection changed 12.12.2024      

def get_video_parameters(cap: cv.VideoCapture):
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    return fps, width, height, frame_count

def print_progress(frame_counter, total_frames, additional_info=""):
    """Выводит прогресс обработки в процентах"""
    percent = (frame_counter / total_frames) * 100
    sys.stdout.write(f"\rProgress: {percent:.1f}% | {additional_info}")
    sys.stdout.flush()

def apply_settings_for_night(bs: cv.BackgroundSubtractorMOG2, frame):
    bs.setBackgroundRatio(settings.NIGHT_BACKGROUND_RATIO)
    # contrast increase for nighttime video
    frame = image_processing.adaptiveHe(frame, contrast=settings.NIGHT_CONTRAST_FACTOR, tile=settings.NIGHT_TILE_GRID)
    mode = "night vision"
    return bs, frame, mode

def apply_settings_for_day(bs: cv.BackgroundSubtractorMOG2, frame):
    bs.setVarThreshold(settings.DAY_VAR_THRESHOLD)
    # correction of gamma (correction of image brightness for daytime - decrease of contrast)
    frame = image_processing.gammac(frame, settings.DAY_GAMMA_TARGET)
    #bs.setBackgroundRatio(0.95)
    mode = 'day light'
    return bs, frame, mode

def set_background_parameters(bs: cv.BackgroundSubtractorMOG2):
    bs.setBackgroundRatio(settings.bs_background_ratio)
    bs.setComplexityReductionThreshold(settings.bs_complexity_reduction_threshold)
    bs.setVarThresholdGen(settings.bs_var_threshold_gen)
 
    bs.setNMixtures(settings.bs_NMixtures)
    bs.setShadowThreshold(settings.bs_shadow_threshold)
    bs.setShadowValue(settings.bs_shadow_value)
    bs.setVarInit(settings.bs_VarInit)
    bs.setVarMax(settings.bs_VarMax)
    bs.setVarMin(settings.bs_VarMin)

    return bs
 
# moving objects detector main method 
def main(input_file=None, output_file=None, output_video=None):
    ''' Detection of moving objects using background subtractor MOG2. 
        Modified to support progress reporting.
    '''
    # Переопределяем настройки, если переданы аргументы
    if input_file:
        settings.inputfile = input_file
    if output_file:
        settings.outputfile = output_file
    
    vcap = cv.VideoCapture(settings.inputfile)
    if not vcap.isOpened():
        print("Cannot open video. Quitting the program.")
        sys.exit()
    
    # Получаем параметры видео
    fps, width, height, total_frames = get_video_parameters(vcap)
    if total_frames <= 0:
        print("Error: Could not determine total frame count.")
        sys.exit()
    
    # Инициализируем VideoWriter для сохранения результата
    if output_video:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # creation of BackgroundSubstractor object 
    bs = cv.createBackgroundSubtractorMOG2(history=settings.bs_history, 
                                         varThreshold=settings.bs_var_threshold, 
                                         detectShadows=settings.bs_detect_shadows) 
    bs = set_background_parameters(bs)
    detector_IO.print_background_parameters(bs)
    detector_IO.print_video_parameters(fps, width, height)
    
    # variables for the detector 
    frame_counter = 0
    t0 = time.perf_counter()
    t1 = t0 + 0.04
    ttime = 0
    
    detections = collections.deque(maxlen=settings.buffer_size)
    detector_IO.clear_output_file(settings.outputfile)
   
    nightMode = False
    print(f"\nProcessing video: {os.path.basename(settings.inputfile)}")
    print(f"Total frames: {total_frames}")
    
    while vcap.isOpened():
        dt = t1 - t0
        fpsp = 1 / dt
        ttime += dt
        
        # Выводим прогресс обработки
        processing_info = f"Frame: {frame_counter}/{total_frames} | Speed: {fpsp:.1f}fps"
        print_progress(frame_counter, total_frames, processing_info)
        
        ret, frame = vcap.read()
        if not ret:
            break
        frame_counter += 1
        
        if frame_counter % settings.check_night_per_frames == 0 or frame_counter == 1:
            nightMode = image_processing.nightVision(frame)
        
        # Обработка кадра (без изменений)
        processing_frame = frame
        mode = "None" 
        if nightMode:
            nf_threshold_id = settings.nf_threshold_night
            bs, processing_frame, mode = apply_settings_for_night(bs, frame)
        else:
            nf_threshold_id = settings.nf_threshold_day 
            bs, processing_frame, mode = apply_settings_for_day(bs, frame)
        
        foreground = bs.apply(processing_frame)

        framedata = [] 
        if frame_counter > settings.frames_skip:
            foreground = image_processing.foreground_postprocessing(foreground)
            contours, hierarchy = cv.findContours(foreground, settings.CONTOUR_RETRIEVAL_MODE, settings.CONTOUR_APPROX_METHOD)
            framedata = detection_processing.detect(framedata, contours)

        outer_detections = detection_processing.remove_nested_detections(framedata, settings.NESTED_DETECTION_OVERLAP_THRESHOLD)
        detections.append(outer_detections)
        detections, nnids, one, two = detection_processing.assignIDs(detections, nf_threshold_id)
        
        detections, nidsc = detection_processing.completeIDs(detections, nf_threshold_id)
        detections, nobsfr, one, two = detection_processing.validateObjs(detections, frame_counter, fps, nf_threshold_id)

        detector_IO.print_frame_detection(settings.outputfile, detections[len(detections)-1], frame_counter)
        frame_with_boxes = detector_IO.show_boxes(frame.copy(), detections[len(detections)-1], fps)
        
        if output_video:
            out.write(frame_with_boxes)
            
        detector_IO.save_frame_with_boxes(frame_with_boxes, frame_counter, int(len(str(total_frames))))
        
        t1 = time.perf_counter()
    
    # Завершаем вывод прогресса
    print_progress(frame_counter, total_frames, "Completed!")
    print(f"\nAverage processing speed: {frame_counter/ttime:.1f} fps")
    
    # Освобождаем ресурсы
    vcap.release()
    if output_video:
        out.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Moving objects detection')
    parser.add_argument('--input', help='Input video file', required=True)
    parser.add_argument('--output', help='Output JSON file', required=True)
    parser.add_argument('--video', help='Output video file with detections', required=True)
    args = parser.parse_args()
    
    # Проверяем существование входного файла
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        sys.exit(1)
        
    # Проверяем доступность папки для выходных файлов
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        main(input_file=args.input, output_file=args.output, output_video=args.video)
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        sys.exit(1)


