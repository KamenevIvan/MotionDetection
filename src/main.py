import cv2 as cv
import math
import time
import sys
import base64
import json
import collections

import settings
import detector_IO
import image_processing
import detection_processing

######## moving objects detector module #################
#   Version 8 from 29.11.2024 detection changed 12.12.2024      

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
    bs.setBackgroundRatio(0.95)
    mode = 'day light'
    return bs, frame, mode

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

    vcap = cv.VideoCapture(settings.inputfile)
    if not vcap.isOpened():
        print("Cannot open video. Quitting the program.")
        sys.exit()
    
    # creation of BackgroundSubstractor object 
    bs = cv.createBackgroundSubtractorMOG2(history = settings.background_history, varThreshold = 30, detectShadows = True) #varThreshold - detection threshold  
    bs = set_background_parameters(bs)
    #detector_IO.print_background_parameters(bs)

    fps, width, height = get_video_parameters(vcap)
    #detector_IO.print_video_parameters(fps, width, height)
    
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
        frame_counter += 1
        
        if frame_counter % settings.check_night_per_frames == 0 or frame_counter == 1:
            nightMode = image_processing.nightVision(frame) # ИЗМЕРИТЬ
        
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


        foreground = bs.apply(processing_frame)
        framedata = [] 
        
        if frame_counter > settings.frames_skip:
            foreground = image_processing.fix_image(foreground)# ИЗМЕРИТЬ
            contours, hierarchy = cv.findContours(foreground, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # ИЗМЕРИТЬ
            framedata = detection_processing.detect(framedata, contours)# ИЗМЕРИТЬ
                
        detections.append(framedata)
        detections, nnids = detection_processing.assignIDs(detections, nf_threshold_id)# ИЗМЕРИТЬ
        detector_IO.print_console(f'Assigned {nnids} new ids')
        
        # downfilling id for objects with just assigned ids for previous frames
        detections, nidsc = detection_processing.completeIDs(detections, nf_threshold_id)# ИЗМЕРИТЬ
        #detector_IO.print_console(f'{nidsc} objects have their id filled in previous frames')
        
        # validate objects in detections for suitability for reporting   
        detections, nobsfr = detection_processing.validateObjs(detections, frame_counter, fps, nf_threshold_id) # ИЗМЕРИТЬ
        detector_IO.print_console(f'{nobsfr} objects were marked suitable for reporting')
        detector_IO.print_console(f'current framedata: {detections[len(detections)-1]}')


        detector_IO.print_frame_detection(settings.outputfile, detections[len(detections)-1], frame_counter)
        frame_with_boxes = detector_IO.show_boxes(frame, detections[len(detections)-1], fps)
        detector_IO.save_frame_with_boxes(frame_with_boxes, frame_counter, int(len(str(vcap.get(cv.CAP_PROP_FRAME_COUNT)))))     
        
        t1 = time.perf_counter()
    vcap.release()
    cv.destroyAllWindows()
    print('Video processing stopped. Average time for frame processing (s):', ttime / frame_counter)
    #############################################################

    
if __name__ == '__main__':
    main()


