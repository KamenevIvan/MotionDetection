import cv2 as cv
import detection_processing
import settings

def print_console(text):
    if settings.enable_print_console:
        print(text)

def print_background_parameters(bs: cv.BackgroundSubtractorMOG2):
    if settings.enable_print_console:
        print("Parameters of the background subtractor MOG2:")
        print(f'background ratio={bs.getBackgroundRatio():.4f}, complexity reduction threshold={bs.getComplexityReductionThreshold():.4f}') 
        print(f'detect shadows={bs.getDetectShadows()}, history={bs.getHistory()}, number of gaussians={bs.getNMixtures()}') 
        print(f'shadow threshold={bs.getShadowThreshold()}, shadow value={bs.getShadowValue()}, initial variance={bs.getVarInit()}') 
        print(f'max variance={bs.getVarMax()}, min variance={bs.getVarMin()}, main threshold={bs.getVarThreshold()}')
        print(f'new gaussian component variance threshold={bs.getVarThresholdGen()}')

def print_video_parameters(fps, width, height):
    if settings.enable_print_console:
        print("Video parameters:")
        print(f'fps = {fps}, width = {width}, height = {height}')

def clear_output_file(path):
    if settings.enable_output_file:
        line = 'Frame#, IDs, xmin, ymin, width, height'
        with open(path, "w") as file: 
            file.write(line)

def print_frame_detection(path, detections: list[detection_processing.Detection], frame_counter):
    if settings.enable_output_file:
        with open(path, "a") as file:
            for detect in detections:
                line = str(frame_counter) + ',' + str(detect.id) + ',' + str(detect.x) + ',' + str(detect.y) + ',' + str(detect.width) + ',' + str(detect.height) + "1,-1,-1,-1\n"
                file.write(line)

def show_boxes(frame, detections: list[detection_processing.Detection], fps):
    if settings.enable_show_boxes or settings.enable_save_frames:
        for detect in detections:
            cv.rectangle(frame, (detect.x, detect.y), (detect.x + detect.width, detect.y + detect.height), settings.box_color, 2)
            if detect.id != -1:
                label = f"ID: {detect.id}"
                text_position = (detect.x, detect.y - 10)
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                font_thickness = 2
                text_color = settings.box_color
                cv.putText(frame, label, text_position, font, font_scale, text_color, font_thickness)

        if settings.enable_show_boxes:
            cv.imshow("Video-zagolovok okna", frame)
            cv.waitKey(int(1000/fps))
            cv.waitKey(1)
        return frame
    return None

def save_frame_with_boxes(frame, frame_counter, leng):
    if settings.enable_save_frames and frame is not None:
        cv.imwrite(settings.output_frame_dir + "/image_with_box"+ str(frame_counter).zfill(leng)+".jpg", frame)
            