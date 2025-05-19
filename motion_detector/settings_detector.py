import cv2 as cv

######## detector settings (global variables) ############

buffer_size = 3000
nf_object = 3000          # maximum number of frames for object to be tracked = length of detections CL 
iou_threshold_id = 0.2     # threshold for iou between consequtive frames for object to tracked (0 .. 1.0) 
scale_threshold_id = 2.0   # threshold scaling factor for bboxes dimensions w and h 
nf_threshold_day = 17      # number of frames object should appear continuously to be registered - day mode (2 ..) 
nf_threshold_night = 17    # number of frames object should appear continuously to be registered - night vision mode (2 ..)  
tr_mov_threshold = 14.0    # translational movement threshold - minimum change in object location (px)  
report_frame = 1         # REPORT EVERY FRAME
area_threshold = 256       # minimum area of objects for detection (px**2)
ntol = 4.0    # threshold difference between blue, green and red channels in night vision mode 
frames_skip = 100
check_night_per_frames = 10
update_background_per_frames = 10

######## Background Substractor Parameters ############################################################################################
#important parameters
bs_background_ratio = 0.9                  # default 0.9, pixel constant for BgR * History is added to background model  ~ background update rate 
bs_complexity_reduction_threshold = 0.05   # default 0.05 - ~number of samples for the component to be accepted to background
bs_var_threshold_gen = 9.0                 # default 9.0, threshold of --||-- of sample to existing background component-above-start new component    
bs_var_threshold = 30.0                    # default 16, main detector threshold - set in constructor  #varThreshold - detection threshold   

# not very important parameters
bs_detect_shadows = True                   # default - set in constructor
bs_history = 200                           # default 500, number of frames that influence background model - set in constructor 
bs_NMixtures = 5                           # default 5, number of gaussians in the background model 
bs_shadow_threshold = 0.5                  # default 0.5, shadows darker than 0.5 are considered as moving objects 
bs_shadow_value = 127                      # default 127, pixel value to mark shadows 
bs_VarInit = 15.0                          # default 15.0, initial variance of each gaussian component
bs_VarMax = 75.0                           # default 75.0
bs_VarMin = 4.0                            # default 4.0
######################################################################################################################################

# Параметры ночного режима
NIGHT_BACKGROUND_RATIO = 0.5         # из apply_settings_for_night()
NIGHT_CONTRAST_FACTOR = 2.0          # из frame = adaptiveHe(..., contrast=2.0)
NIGHT_TILE_GRID = (8, 8)             # из adaptiveHe(..., tile=(8,8))

# Параметры дневного режима 
DAY_VAR_THRESHOLD = 200.0            # из apply_settings_for_day()
DAY_GAMMA_TARGET = 150               # из gammac(..., tavg=155.0)

# Морфологические операции
EROSION_KERNEL_SIZE = (3, 3)         # из cv.getStructuringElement(2, (3,3))
DILATION_KERNEL_SIZE = (10, 20)      # из cv.getStructuringElement(2, (10,20))
DILATION_ITERATIONS = 2              # из cv.dilate(..., iterations=2)

# CLAHE параметры
ADAPTIVE_HE_CONTRAST = 2.0           # из createCLAHE(clipLimit=2.0)
ADAPTIVE_HE_TILE_GRID = (8, 8)       # из tileGridSize=(8,8)

# Обработка контуров
NESTED_DETECTION_OVERLAP_THRESHOLD = 0.6  # из remove_nested_detections(..., 0.6)
CONTOUR_RETRIEVAL_MODE = cv.RETR_EXTERNAL  # из findContours(..., cv.RETR_EXTERNAL)
CONTOUR_APPROX_METHOD = cv.CHAIN_APPROX_SIMPLE  # из findContours(..., cv.CHAIN_APPROX_SIMPLE)

# Визуализация
BOX_LINE_THICKNESS = 2               # из cv.rectangle(..., thickness=2)
TEXT_FONT_SCALE = 0.8                # из cv.putText(..., font_scale=0.8)
TEXT_FONT_THICKNESS = 2              # из cv.putText(..., thickness=2)




##########################      ПАРАМЕТРЫ ЗАПУСКА          ############################################


enable_print_console = False
enable_output_file = False
enable_show_boxes = True
enable_save_frames = False

inputfile = "PolarBear3.mp4"  
outputfile = 'mot-results.txt'
output_frame_dir = r"F:\VScode\NSU\DetectMotion\VKI_CAMERAS_TESTS\magadan\frames"
box_color = (0, 0, 255)

if buffer_size < nf_object:
    buffer_size = nf_object