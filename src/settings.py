######## detector settings (global variables) ############

buffer_size = 30
nf_object = (
    3000  # maximum number of frames for object to be tracked = length of detections CL
)
background_history = 200  # number of frames in background subtractor history
iou_threshold_id = (
    0.2  # threshold for iou between consequtive frames for object to tracked (0 .. 1.0)
)
scale_threshold_id = 2.0  # threshold scaling factor for bboxes dimensions w and h
nf_threshold_day = 17  # number of frames object should appear continuously to be registered - day mode (2 ..)
nf_threshold_night = 17  # number of frames object should appear continuously to be registered - night vision mode (2 ..)
tr_mov_threshold = (
    14.0  # translational movement threshold - minimum change in object location (px)
)
report_frame = 1  # REPORT EVERY FRAME
area_threshold = 256  # minimum area of objects for detection (px**2)
ntol = 4.0  # threshold difference between blue, green and red channels in night vision mode
frames_skip = 100
check_night = 10

enable_print_console = False
enable_output_file = False
enable_show_boxes = False
enable_save_frames = False

inputfile = "input.mp4"
outputfile = "mot-results.txt"
output_frame_dir = "frames_out"
box_color = (0, 255, 0)

if buffer_size < nf_object:
    buffer_size = nf_object
