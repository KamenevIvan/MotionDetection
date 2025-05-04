import cv2 as cv
import math
import time
import sys
import base64
import json
import collections
import optuna
from functools import partial

import settings
import detector_IO
import image_processing
import detection_processing

######## moving objects detector module #################
#   Version 8 from 29.11.2024 detection changed 12.12.2024      

class ParameterOptimizer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Cannot open video for optimization")
            
        self.fps, self.width, self.height = self._get_video_parameters()
        
    def _get_video_parameters(self):
        fps = self.cap.get(cv.CAP_PROP_FPS)
        width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        return fps, width, height
        
    def evaluate_params(self, params, max_frames=50):
        """Evaluate detector parameters on sample frames"""
        if params['color_space'] == 'HSV':
            detector = image_processing.BackgroundSubtractor_HSV(
                buffer_size=params['buffer_size'],
                frame_width=self.width,
                frame_height=self.height
            )
            detector.threshold_h = params['threshold_h']
            detector.threshold_s = params['threshold_s']
            detector.threshold_v = params['threshold_v']
        elif params['color_space'] == 'YCbCr':
            detector = image_processing.BackgroundSubtractor_YCbCr(
                buffer_size=params['buffer_size'],
                frame_width=self.width,
                frame_height=self.height
            )
            detector.threshold_y = params['threshold_y']
            detector.threshold_chroma = params['threshold_chroma']
        else:  # GRAY
            detector = image_processing.BackgroundSubtractor(
                buffer_size=params['buffer_size'],
                frame_width=self.width,
                frame_height=self.height
            )
            detector.threshold_value = params['threshold']
            
        detector.blur = params['blur']
        detector.minimum_are_contours = params['min_area']

        merge_distance_threshold = params.get('merge_distance_threshold', 5)
        overlap_threshold = params.get('overlap_threshold', 0.6)
        
        total_objects = 0
        processing_time = 0
        frame_count = 0
        
        while frame_count < max_frames:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            start_time = time.time()
            
            detector.add_frame(frame)
            detector.create_background()
            foreground = detector.extract_foreground(frame)
            
            contours, _ = cv.findContours(foreground, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            framedata = []
            for contour in contours:
                area = cv.contourArea(contour)
                if area > params['min_area']:
                    x, y, w, h = cv.boundingRect(contour)
                    detection = detection_processing.Detection(
                        id=-1, x=x, y=y, width=w, height=h, 
                        vx=0, vy=0, nf=1, indxprev=-1, sfr=False, fnrlt=-1
                    )
                    framedata.append(detection)
            
            merged_detections = detection_processing.merge_close_detections(framedata, merge_distance_threshold)
            final_detections = detection_processing.remove_nested_detections(merged_detections, overlap_threshold)
            
            processing_time += time.time() - start_time
            total_objects += len(final_detections)
            frame_count += 1

        self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)

        avg_objects = total_objects / frame_count if frame_count > 0 else 0
        avg_time = processing_time / frame_count if frame_count > 0 else 0

        score = avg_objects / (1 + avg_time)
        return score
        
    def optimize(self, n_trials=100):
        """Optimize detector parameters using Optuna"""
        study = optuna.create_study(direction='maximize')
        study.optimize(partial(self._objective), n_trials=n_trials)
        return study.best_params
        
    def _objective(self, trial):
        """Objective function for Optuna"""
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
            
        return self.evaluate_params(params)

def set_background_parameters(bs: cv.BackgroundSubtractorMOG2):
    bs.setBackgroundRatio(0.9)              # default 0.9, pixel constant for BgR * History is added to background model  ~ background update rate 
    bs.setComplexityReductionThreshold(0.05)   # default 0.05 - ~number of samples for the component to be accepted to background
    bs.setVarThresholdGen(9.0)     # default 9.0, threshold of --||-- of sample to existing background component-above-start new component    
    #bs.setVarThreshold(45.0)      # default 16, main detector threshold - set in constructor   

    #bs.setDetectShadows(True)      # default - set in constructor
    #bs.setHistory(200)             # default 500, number of frames that influence background model - set in constructor 
    bs.setNMixtures(5)             # default 5, number of gaussians in the background model 
    bs.setShadowThreshold(0.5)     # default 0.5, shadows darker than 0.5 are considered as moving objects 
    bs.setShadowValue(127)         # default 127, pixel value to mark shadows 
    bs.setVarInit(15.0)            # default 15.0, initial variance of each gaussian component
    bs.setVarMax(75.0)             # default 75.0
    bs.setVarMin(4.0)              # default 4.0

    return bs

def get_video_parameters(cap: cv.VideoCapture):
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    return fps, width, height

def apply_settings_for_night(bs: cv.BackgroundSubtractorMOG2, frame):
    bs.setBackgroundRatio(0.5)
    # contrast increase for nighttime video
    frame = image_processing.adaptiveHe(frame, contrast=2.0, tile=(8,8))
    mode = "night vision"
    return bs, frame, mode

def apply_settings_for_day(bs: cv.BackgroundSubtractorMOG2, frame):
    bs.setVarThreshold(200.0)
    # correction of gamma (correction of image brightness for daytime - decrease of contrast)
    frame = image_processing.gammac(frame, 150)
    #bs.setBackgroundRatio(0.95)
    mode = 'day light'
    return bs, frame, mode

def unpacking(listOfLists):
    return [item for sublist in listOfLists for item in sublist] 

def resize_to_fit(image, max_height, max_width):
    height, width = image.shape[:2]
    scale = min(max_height / height, max_width / width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv.resize(image, (new_width, new_height))

def load_optimized_params():
    """Load optimized parameters from file if exists"""
    try:
        with open('optimized_params.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def save_optimized_params(params):
    """Save optimized parameters to file"""
    with open('optimized_params.json', 'w') as f:
        json.dump(params, f, indent=2)

# moving objects detector main method 
def main():
    ''' Detection of moving objects using background subtractor MOG2. 
        Version 7 created 28.11.2024  
        Reports: new moving objects, existing moving objects every report_period seonds
        Report format: [[x0, y0, x1, y1, trackID, classID, confidence, bestcropBase64], ...]             
        Input: 
        Output: json files with results recorded every report_period seconds for every object detected 
        Functions required: nightVision(), gammac(), adaptiveHe(), putElementCL(), 
                            assignIDs(), completeIDs(), validateObjs(), getIDsfr()                             
    '''

    if settings.ENABLE_ANNOTATION_OPTIMIZATION:
        print("Running parameter optimization...")
        optimizer = ParameterOptimizer(settings.inputfile)
        optimized_params = optimizer.optimize(n_trials=settings.OPTIMIZATION_TRIALS)
        save_optimized_params(optimized_params)
        print("Optimization complete. Best parameters:", optimized_params)
    else:
        optimized_params = load_optimized_params()
        if optimized_params:
            print("Loaded optimized parameters:", optimized_params)

    merge_distance_threshold = optimized_params.get('merge_distance_threshold', 5) if optimized_params else 5
    overlap_threshold = optimized_params.get('overlap_threshold', 0.6) if optimized_params else 0.6

    nightModeTimeMes = []
    fixImgTimeMes = []
    findContersTimeMes = []
    detectTimeMes = []
    assignIDTimeMes = []
    completeIDTimeMes = []
    validateObjTimeMes = []
    applyTimeMes = []
    BackForeTimeMes = []
    trajdiamTimeMes = []
    trajdiamOldTimeMes = []
    relSiouTimeMes = []
    relSiouOldTimeMes = []

    BackForeTimeMes_fast = []
    BackForeTimeMes_YCrCb = []
    BackForeTimeMes_HSV = []

    vcap = cv.VideoCapture(settings.inputfile)
    if not vcap.isOpened():
        print("Cannot open video. Quitting the program.")
        sys.exit()
    
    # creation of BackgroundSubstractor object 
    bs = cv.createBackgroundSubtractorMOG2(history = settings.background_history, varThreshold = 30, detectShadows = True) #varThreshold - detection threshold  
    bs = set_background_parameters(bs)
    #detector_IO.print_background_parameters(bs)

    fps, width, height = get_video_parameters(vcap)
    #bs = None
    bs_gray = image_processing.BackgroundSubtractor(settings.background_history, width, height)
    bs_fast = image_processing.BackgroundSubtractor_Fast(settings.background_history, width, height)
    bs_YCrCb = image_processing.BackgroundSubtractor_YCbCr(settings.background_history, width, height)
    bs_hsv = image_processing.BackgroundSubtractor_HSV(settings.background_history, width, height)
    
    # Apply optimized parameters if available
    if optimized_params:
        if optimized_params['color_space'] == 'HSV':
            bs_hsv.threshold_h = optimized_params.get('threshold_h', bs_hsv.threshold_h)
            bs_hsv.threshold_s = optimized_params.get('threshold_s', bs_hsv.threshold_s)
            bs_hsv.threshold_v = optimized_params.get('threshold_v', bs_hsv.threshold_v)
            bs_hsv.blur = optimized_params.get('blur', bs_hsv.blur)
            bs_hsv.minimum_are_contours = optimized_params.get('min_area', bs_hsv.minimum_are_contours)
        elif optimized_params['color_space'] == 'YCbCr':
            bs_YCrCb.threshold_y = optimized_params.get('threshold_y', bs_YCrCb.threshold_y)
            bs_YCrCb.threshold_chroma = optimized_params.get('threshold_chroma', bs_YCrCb.threshold_chroma)
            bs_YCrCb.blur = optimized_params.get('blur', bs_YCrCb.blur)
            bs_YCrCb.min_contour_area = optimized_params.get('min_area', bs_YCrCb.min_contour_area)
        else:  # GRAY
            bs_gray.threshold_value = optimized_params.get('threshold', bs_gray.threshold_value)
            bs_gray.blur = optimized_params.get('blur', bs_gray.blur)
            bs_gray.minimum_are_contours = optimized_params.get('min_area', bs_gray.minimum_are_contours)
    
    # variables for the detector 
    frame_counter = 0                               # frame counter
    # execution time measurement variables  
    t0 = 0                               # time of start of processing cycle for each frame  
    t1 = 0.04                            # time of finishing processing each frame 
    ttime = 0                            # total processing time of frames   
    
    detections = collections.deque(maxlen=settings.buffer_size)
    detector_IO.clear_output_file(settings.outputfile)
   
    nightMode = False
    while vcap.isOpened():
        # execution time and fps for previous frame
        dt = t1 - t0          # time for processing previous frame
        fpsp = 1 / dt         # fps for video processing 
        ttime += dt           # total time 
        detector_IO.print_console(f'processing speed fps: {fpsp}')
        t0 = time.perf_counter()
                
        ret, frame = vcap.read()
        if not ret:
            print("Cannot read video frame. Video stream may have ended. Exiting.")
            break
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_counter += 1
        
        if frame_counter % settings.check_night_per_frames == 0 or frame_counter == 1:
            start = time.time()
            nightMode = image_processing.nightVision(frame) # ИЗМЕРИТЬ
            nightModeTimeMes.append(time.time()-start)

        
        #### preprocessing dependent on the scene conditions (day, night, rain, snow, snowfall)
        
        processing_frame = frame
        mode = "None" 
        if nightMode:
            nf_threshold_id = settings.nf_threshold_night
            bs, processing_frame, mode = apply_settings_for_night(bs, frame)
        else:
            nf_threshold_id = settings.nf_threshold_day 
            bs, processing_frame, mode = apply_settings_for_day(bs, frame)
        detector_IO.print_console(f'Detected camera mode: {mode}')     
        
        start = time.time()
        foreground = bs.apply(processing_frame)
        applyTimeMes.append(time.time()-start)

        bs_gray.add_frame(processing_frame)
        bs_gray.create_background()
        start = time.time()
        foreground = bs_gray.extract_foreground(processing_frame)
        BackForeTimeMes.append(time.time()-start)

        bs_fast.add_frame(processing_frame)
        bs_fast.create_background()
        start = time.time()
        foreground_fast = bs_fast.extract_foreground(processing_frame)
        BackForeTimeMes_fast.append(time.time()-start)

        bs_YCrCb.add_frame(processing_frame)
        bs_YCrCb.create_background()
        start = time.time()
        foreground_ycrcb = bs_YCrCb.extract_foreground(processing_frame)
        BackForeTimeMes_YCrCb.append(time.time()-start)

        bs_hsv.add_frame(processing_frame)
        bs_hsv.create_background()
        start = time.time()
        foreground_hsv = bs_hsv.extract_foreground(processing_frame)
        BackForeTimeMes_HSV.append(time.time()-start)

        framedata = [] 
        
        if frame_counter > settings.frames_skip:
            start = time.time()
            foreground = image_processing.fix_image_old(foreground)# ИЗМЕРИТЬ
            foreground_ = image_processing.fix_image(foreground)# ИЗМЕРИТЬ
            fixImgTimeMes.append(time.time()-start)

            start = time.time()
            contours, hierarchy = cv.findContours(foreground, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # ИЗМЕРИТЬ
            findContersTimeMes.append(time.time()-start)

            start = time.time()
            framedata = detection_processing.detect(framedata, contours)# ИЗМЕРИТЬ
            detectTimeMes.append(time.time()-start)

        merged_detections = detection_processing.merge_close_detections(framedata, merge_distance_threshold)
        outer_detections = detection_processing.remove_nested_detections(merged_detections, overlap_threshold)
        
        detections.append(outer_detections)
        start = time.time()
        detections, nnids, one, two = detection_processing.assignIDs(detections, nf_threshold_id)# ИЗМЕРИТЬ
        assignIDTimeMes.append(time.time()-start)
        relSiouTimeMes.append(one)
        relSiouOldTimeMes.append(two)

        detector_IO.print_console(f'Assigned {nnids} new ids')
        
        # downfilling id for objects with just assigned ids for previous frames
        start = time.time()
        detections, nidsc = detection_processing.completeIDs(detections, nf_threshold_id)# ИЗМЕРИТЬ
        completeIDTimeMes.append(time.time()-start)

        #detector_IO.print_console(f'{nidsc} objects have their id filled in previous frames')
        
        # validate objects in detections for suitability for reporting
        start = time.time()   
        detections, nobsfr, one, two = detection_processing.validateObjs(detections, frame_counter, fps, nf_threshold_id) # ИЗМЕРИТЬ
        validateObjTimeMes.append(time.time()-start)
        trajdiamTimeMes.append(one)
        trajdiamOldTimeMes.append(two) 

        detector_IO.print_console(f'{nobsfr} objects were marked suitable for reporting')
        detector_IO.print_console(f'current framedata: {detections[len(detections)-1]}')

        detector_IO.print_frame_detection(settings.outputfile, detections[len(detections)-1], frame_counter)
        frame_with_boxes = detector_IO.show_boxes(frame.copy(), detections[len(detections)-1], fps)
        detector_IO.save_frame_with_boxes(frame_with_boxes, frame_counter, int(len(str(vcap.get(cv.CAP_PROP_FRAME_COUNT)))))     
        
        t1 = time.perf_counter()
    vcap.release()
    cv.destroyAllWindows()
    print('Video processing stopped. Average time for frame processing (s):', ttime / frame_counter)
    with open("timeResults.txt", "w") as file:  
        file.write(f"""
        OpenCV: Avg: {sum(applyTimeMes) / len(applyTimeMes):.5f}, Max: {max(applyTimeMes):.5f} | {applyTimeMes}\n
        Custom Gray_Fast: Avg: {sum(BackForeTimeMes_fast) / len(BackForeTimeMes_fast):.5f}, Max: {max(BackForeTimeMes_fast):.5f} | {BackForeTimeMes_fast}\n
        Custom Gray: Avg: {sum(BackForeTimeMes) / len(BackForeTimeMes):.5f}, Max: {max(BackForeTimeMes):.5f} | {BackForeTimeMes}\n
        Custom YCbCr: Avg: {sum(BackForeTimeMes_YCrCb) / len(BackForeTimeMes_YCrCb):.5f}, Max: {max(BackForeTimeMes_YCrCb):.5f} | {BackForeTimeMes_YCrCb}\n
        Custom HSV: Avg: {sum(BackForeTimeMes_HSV) / len(BackForeTimeMes_HSV):.5f}, Max: {max(BackForeTimeMes_HSV):.5f} | {BackForeTimeMes_HSV}\n
        """)

if __name__ == '__main__':
    main()
