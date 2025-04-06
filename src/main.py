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

def get_video_parameters(cap: cv.VideoCapture):
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    return fps, width, height

def apply_settings_for_night(frame):
    # contrast increase for nighttime video
    frame = image_processing.adaptiveHe(frame, contrast=2.0, tile=(8,8))
    mode = "night vision"
    return frame, mode

def apply_settings_for_day(frame):

    # correction of gamma (correction of image brightness for daytime - decrease of contrast)
    frame = image_processing.gammac(frame, 150)
    mode = 'day light'
    return frame, mode

def unpacking(listOfLists):
    return [item for sublist in listOfLists for item in sublist] 

def resize_to_fit(image, max_height, max_width):
    height, width = image.shape[:2]
    scale = min(max_height / height, max_width / width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv.resize(image, (new_width, new_height))

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

    fps, width, height = get_video_parameters(vcap)
    bs = image_processing.BackgroundSubtractor(settings.background_history, width, height)
    
    # variables for the detector 
    frame_counter = 0                     # frame counter
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
            nightMode = image_processing.nightVision(frame)

        
        #### preprocessing dependent on the scene conditions (day, night, rain, snow, snowfall)
        
        processing_frame = frame
        mode = "None" 
        if nightMode:
            nf_threshold_id = settings.nf_threshold_night
            processing_frame, mode = apply_settings_for_night(frame)
        else:
            nf_threshold_id = settings.nf_threshold_day 
            processing_frame, mode = apply_settings_for_day(frame)
        detector_IO.print_console(f'Detected camera mode: {mode}')     
        
        bs.add_frame(processing_frame)
        bs.create_background()
        foreground = bs.extract_foreground(processing_frame)


        framedata = [] 
        
        if frame_counter > settings.frames_skip:
            foreground = image_processing.fix_image(foreground)
            contours, hierarchy = cv.findContours(foreground, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            framedata = detection_processing.detect(framedata, contours)

                
        sorted_detections = detection_processing.remove_nested_detections(framedata, 0.8)
        merged_detections = detection_processing.merge_close_detections(sorted_detections, 10)

        detections.append(merged_detections)

        detections, nnids, one, two = detection_processing.assignIDs(detections, nf_threshold_id)
        detector_IO.print_console(f'Assigned {nnids} new ids')
        
        # downfilling id for objects with just assigned ids for previous frames
        detections, nidsc = detection_processing.completeIDs(detections, nf_threshold_id)# ИЗМЕРИТЬ
        detector_IO.print_console(f'{nidsc} objects have their id filled in previous frames')
        
        # validate objects in detections for suitability for reporting   
        detections, nobsfr, one, two = detection_processing.validateObjs(detections, frame_counter, fps, nf_threshold_id) # ИЗМЕРИТЬ) 
        detector_IO.print_console(f'{nobsfr} objects were marked suitable for reporting')
        detector_IO.print_console(f'current framedata: {detections[len(detections)-1]}')

        detector_IO.print_frame_detection(settings.outputfile, detections[len(detections)-1], frame_counter)
        frame_with_boxes = detector_IO.show_boxes(frame.copy(), detections[len(detections)-1], fps)
        detector_IO.save_frame_with_boxes(frame_with_boxes, frame_counter, int(len(str(vcap.get(cv.CAP_PROP_FRAME_COUNT)))))     
        
        t1 = time.perf_counter()
    vcap.release()
    cv.destroyAllWindows()
    print('Video processing stopped. Average time for frame processing (s):', ttime / frame_counter)
    #############################################################Assign IDs: Avg: {sum(assignIDTimeMes) / len(assignIDTimeMes):.5f}, Max: {max(assignIDTimeMes):.5f} | {assignIDTimeMes}\n

    
if __name__ == '__main__':
    main()


