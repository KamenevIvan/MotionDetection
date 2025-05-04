######## detector settings (global variables) ############

nf_object = 3000          # maximum number of frames for object to be tracked = length of detections CL 
background_history = 100    # number of frames in background subtractor history 
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

enable_print_console = False
enable_output_file = True
enable_show_boxes = True
enable_save_frames = False

ENABLE_ANNOTATION_OPTIMIZATION = False
IOU_THRESHOLD = 0.5

inputfile = r"C:\uni\project\TLP\PolarBear3\PolarBear3.mp4"
outputfile = 'mot-results.txt'
output_frame_dir = r"C:\uni\project\MotionDetection\VKI_CAMERAS_TESTS"
box_color = (0, 0, 255)

######## optimized parameters ############

color_space = 'YCbCr'
threshold_y = 10
threshold_chroma = 38
blur = 15
minimum_are_contours = 60
merge_distance_threshold = 11
overlap_threshold = 0.322082618487802
buffer_size = 3000
