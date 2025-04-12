import cv2 as cv
import settings
import math
import numpy as np

import time

class Detection:
    def __init__(self, id, x, y, width, height, vx, vy, nf, indxprev, sfr, fnrlt):
        self.id = id
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.vx = vx
        self.vy = vy
        self.frames_count = nf
        self.indx_prew = indxprev
        self.for_report = sfr
        self.last_frame = fnrlt
# properties - id, x, y, w, h, vx, vy, nf, indxprev, sfr, fnrlt
# nf - number of frames object is present, indxprev - index of object in the prev frame, sfr - if suitable for report
# fnrlt - frame number when object was reported last time

def merge_close_detections(detections: list[Detection], distance_threshold):
    """
    Объединяет детекции, если расстояние между их границами <= threshold.
    Все параметры новой детекции берутся из самой большой детекции в группе.
    """
    if not detections:
        return []
    merged = []
    
    for i, current in enumerate(detections):
        if current is None:
            continue
            
        to_merge = [current]
        
        curr_l, curr_r = current.x, current.x + current.width
        curr_t, curr_b = current.y, current.y + current.height
        
        # Ищем близкие детекции
        for j in range(i+1, len(detections)):
            other = detections[j]
            if other is None:
                continue
                
            other_l, other_r = other.x, other.x + other.width
            other_t, other_b = other.y, other.y + other.height
            
            dx = max(curr_l - other_r, other_l - curr_r)
            dy = max(curr_t - other_b, other_t - curr_b)
            
            if max(dx, dy) <= distance_threshold:
                to_merge.append(other)
                detections[j] = None
        
        if len(to_merge) > 1:
            largest_det = max(to_merge, key=lambda d: d.width * d.height)
            
            new_l = min(d.x for d in to_merge)
            new_r = max(d.x + d.width for d in to_merge)
            new_t = min(d.y for d in to_merge)
            new_b = max(d.y + d.height for d in to_merge)
            
            merged_det = Detection(
                id=largest_det.id,
                x=new_l,
                y=new_t,
                width=new_r - new_l,
                height=new_b - new_t,
                vx=largest_det.vx,
                vy=largest_det.vy,
                nf=largest_det.frames_count,
                indxprev=largest_det.indx_prew,
                sfr=largest_det.for_report,
                fnrlt=largest_det.last_frame
            )
            merged.append(merged_det)
        else:
            merged.append(current)
    
    return merged

def remove_nested_detections(detections: list[Detection], overlap_threshold: float):
    """
    Удаляет детекции, которые почти полностью находятся внутри других (больших) детекций,
    используя функцию relSiou для вычисления относительной площади пересечения.
    """
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d.width * d.height, reverse=True)
    outer_detections = []
    
    for current_det in sorted_dets:
        is_nested = False
        x11, y11 = current_det.x, current_det.y
        x12, y12 = x11 + current_det.width, y11 + current_det.height
        
        for outer_det in outer_detections:
            x21, y21 = outer_det.x, outer_det.y
            x22, y22 = x21 + outer_det.width, y21 + outer_det.height
            
            Srel = relSiou(x11, y11, x12, y12, x21, y21, x22, y22)
            if Srel >= overlap_threshold:
                is_nested = True
                break
        
        if not is_nested:
            outer_detections.append(current_det)
    
    return outer_detections

def detect(framedata: list, contours):
    # obtaining bounding upright rectangles 
    for contour in contours:
            
        area = cv.contourArea(contour)
        if area > settings.area_threshold:
            idn = -1      
            x,y,w,h = cv.boundingRect(contour)
            vx = 0
            vy = 0
            nf = 1                # number of frames object is present
            indxprev = -1         # index of object in previous frame if present
            sfr = False           # if object is suitable for reporting
            fnrlt = -1            # frame index of latest reporting of object
            objectdata = Detection(idn, x, y, w, h, vx, vy, nf, indxprev, sfr, fnrlt)
            framedata.append(objectdata)
    return framedata

def relSiou(x11, y11, x12, y12, x21, y21, x22, y22): # измерить, сравнить время и результаты со старой версией
    ''' Computation of relative area of intersection between two rectangles 
        relSiou = Siou / min(S1, S2)
        Input: coordinates of left top and bottom right corners of the two rectangles 
        Output: relative intersection area in the range of 0 .. 1
    '''
    # Calculate the coordinates of the intersection rectangle
    x_overlap = max(0, min(x12, x22) - max(x11, x21))
    y_overlap = max(0, min(y12, y22) - max(y11, y21))
    
    # Area of intersection
    Siou = x_overlap * y_overlap
    
    # Areas of the two rectangles
    s1 = (x12 - x11) * (y12 - y11)
    s2 = (x22 - x21) * (y22 - y21)
    
    # Relative intersection area
    Srel = Siou / min(s1, s2) if min(s1, s2) != 0 else 0
    
    return Srel

def relSiou_old(x11, y11, x12, y12, x21, y21, x22, y22):  # измерить, сравнить время и результаты с новой версией
    ''' Computation of relative area of intersection between two rectangles 
        relSiou = Siou / min(S1, S2)
        Input: coordinates of left top and bottom right corners of the two rectangles 
        Output: relative intersection area in the range of 0 .. 1
    '''
    # dimensions of the intersection rectangle 
    lx = 0
    ly = 0
    # width of overlap rectangle 
    # case 1: .|. |
    if x21 <= x11 and x22 >= x11 and x22 <= x12:
        lx = x22 - x11
    # case 2: .||.
    if x21 <= x11 and x22 >= x12:
        lx = x12 - x11
    # case 3: |..|
    if x21 >= x11 and x21 <= x12 and x22 >= x11 and x22 <= x12:
        lx = x22 - x21
    # case 4: |.|.
    if x21 >= x11 and x21 <= x12 and x22 >= x12:
        lx = x12 - x21
    # height of intersection rectangle 
    # case 1: 
    if y21 <= y11 and y22 >= y11 and y22 <= y12:
        ly = y22 - y11
    # case 2: .||.
    if y21 <= y11 and y22 >= y12:
        ly = y12 - y11
    # case 3: |..|
    if y21 >= y11 and y21 <= y12 and y22 >= y11 and y22 <= y12:
        ly = y22 - y21
    # case 4: |.|.
    if y21 >= y11 and y21 <= y12 and y22 >= y12:
        ly = y12 - y21 
    Siou = lx * ly 
    S1 = (x12 - x11) * (y12 - y11) 
    S2 = (x22 - x21) * (y22 - y21) 
    Srel = Siou / min(S1, S2)      
    return Srel

# detections processing methods 

def assignIDs(detections: list[list[Detection]], nf_threshold_id): 
    ''' This function assigns IDs to objects that are persistently present in the scene for
        nf_threshold_id frames. IDs are checked for uniqueness. 
        Version 1. Changed 06.11.2024, 08.11.2024, 22.11.2024, 28.11.2024, 29.11.2024 
                
        Input: detections - circular list (CL) of lists of lists - 
                 detections[frameindex][objectindex][propertyindex]
                 properties:  id, x, y, w, h, vx, vy, nf, indxprev, sfr, fnrlt
                 ID = -1 for unassigned objects, nf - number of frames of continuous 
                 detection (1..), indxprev - object index in previous frame, sfr - suitable for reporting (True/False)
                 fnrlt - frame index when reported last time (-1 - never reported) 
               nf_threshold_id - number of frames of continuous detection for ID assignment
               dcn - index of the current frame in the detections

        Returns: detections - with assigned ID and fn for the last frame objects 
        
        Global parameters: 
            settings.iou_threshold_id - relSiou threshold  
            scale_threshold_id - scaling threshold
        Требуемые функции:  relSiou() 
    '''
    relSiouTimeMes = []
    relSiouOldTimeMes = []

    dcn = len(detections)-1
    # assigning the attribute id initial value 
    if not hasattr(assignIDs, 'id'):
        assignIDs.id = 0
    
    # checking if object was present in the scene for nf_threshold_id previous frames 
    # criteria for id asignment:  
    #    bboxes of object in consecutive frames should have intersection 
    #      not less than settings.iou_threshold_id
    #    bbox dimensions in consecutive frames - w, h - should be different no more than
    #      scale_threshold_id   
    # among bboxes satisfying the criteria, one is selected that:   has smallest
    #   difference in x and y, in w and h,   -> largest relSiou,    
    #
    nnids = 0   # number of new ids found 
    framedatap = detections[dcn - 1] if dcn > 0 else []    
    # Перебор объектов в текущем кадре
    for current_detection in detections[dcn]:
        # Индекс ближайшего объекта в предыдущем кадре
        coi = -1
        # Наибольшее значение relSiou для ближайшего объекта
        lrelSiou = 0

        # Координаты и размеры текущего объекта
        x11, y11 = current_detection.x, current_detection.y
        w1, h1 = current_detection.width, current_detection.height
        x12, y12 = x11 + w1, y11 + h1

        # Поиск ближайшего объекта в предыдущем кадре
        for j, prev_detection in enumerate(framedatap):
            x21, y21 = prev_detection.x, prev_detection.y
            w2, h2 = prev_detection.width, prev_detection.height
            x22, y22 = x21 + w2, y21 + h2

            # Вычисление relSiou и масштабирования
            start = time.time()
            relSiou12 = relSiou(x11, y11, x12, y12, x21, y21, x22, y22)
            relSiouTimeMes.append(time.time()-start)

            start = time.time()
            relSiou12Test = relSiou_old(x11, y11, x12, y12, x21, y21, x22, y22)
            relSiouOldTimeMes.append(time.time()-start)

            scalew12 = w1 / w2 if w1 / w2 >= 1 else w2 / w1
            scaleh12 = h1 / h2 if h1 / h2 >= 1 else h2 / h1

            # Проверка условий для назначения ID
            if (relSiou12 >= settings.iou_threshold_id and
                scalew12 <= settings.scale_threshold_id and
                scaleh12 <= settings.scale_threshold_id):
                if relSiou12 > lrelSiou:
                    coi = j
                    lrelSiou = relSiou12

        # Если найден близкий объект в предыдущем кадре
        if coi != -1:
            # Назначение индекса ближайшего объекта в предыдущем кадре
            current_detection.indx_prew = coi
            # Увеличение счётчика кадров непрерывного обнаружения
            current_detection.frames_count = framedatap[coi].frames_count + 1

            # Продолжение ID из предыдущего кадра, если он был назначен
            if framedatap[coi].id != -1:
                current_detection.id = framedatap[coi].id
                current_detection.for_report = framedatap[coi].for_report
            else:
                # Назначение нового ID, если счётчик кадров превышает порог
                if current_detection.frames_count >= nf_threshold_id:
                    assignIDs.id += 1
                    current_detection.id = assignIDs.id
                    nnids += 1

        # Если близкий объект в предыдущем кадре не найден
        if coi == -1:
            current_detection.indx_prew = -1

    # Проверка на дубликаты ID в текущем кадре
    ids_in_frame = []
    for detection in detections[dcn]:
        if detection.id == -1:
            continue
        if detection.id in ids_in_frame:
            # Назначение нового ID, если обнаружен дубликат
            assignIDs.id += 1
            detection.id = assignIDs.id
            detection.frames_count = 1  # Сброс счётчика кадров
            detection.indx_prew = -1  # Сброс индекса предыдущего объекта
        else:
            ids_in_frame.append(detection.id)

    return detections, nnids, relSiouTimeMes,  relSiouOldTimeMes


def completeIDs(detections: list[list[Detection]], nf_threshold_id):
    ''' Method to fill object IDs for newly detected objects in the preceding nf_threshold_id - 1 
        frames data
        
        Parameters: detections - circular list of lists of lists
                    dcn - current index of frame in detections 
                    nf_threshold_id - criterion on the number of frames for assigning ID
        Returns: detections with id downfilled for objects with nf = nf_threshold_id, 
                 number of objects id were downfilled for
    '''
    nidsc = 0   # number of ids completed 
    dcn = dcn = len(detections)-1
    # find new objects in the current frame 
    for current_detection  in detections[dcn]: 
        if current_detection.frames_count == nf_threshold_id: 
            idn = current_detection.id
            indxprev = current_detection.indx_prew
            dpn = dcn

            for j in range(nf_threshold_id - 1): 
                # get previous frame index
                dpn -= 1
                detections[dpn][indxprev].id = idn
                indxprev = detections[dpn][indxprev].indx_prew
            nidsc += 1
    return detections, nidsc 
        

def obtainTrajs(detections: list[list[Detection]], dcn: int, ids): 
    ''' A function to obtain trajectories for objects in the video frame   
        Version 2 from 26.11.2024, 29.11.2024 
        parameters: detections - circular list of lists of lists
                    dcn - index of the current frame in detections
                    ids - list of object ids for which obtain trajectories 
        return: dictionary of id and list of points  
    '''
    #print(f'Entered function obtainTrajs(detections, dcn={dcn}, ids={ids}) ')
    #print(f'current frame framedata: {detections[dcn]}')
    # dictionary with object id as key and list of points as value 
    trajs = {}
    for idn in ids:
        nf = -1
        pnts = []  # Initialize pnts inside the loop
        # find object in the current framedata and record its frames count 
        for current_detection in detections[dcn]:
            if current_detection.id == idn:
                nf = current_detection.frames_count  # number of frames the object with id has been detected
                # placing last trajectory point into list of points 
                x = current_detection.x
                y = int(current_detection.y + current_detection.height / 2) 
                pnt = [x, y]
                pnts.append(pnt.copy())
                break
        # looking through frames and placing points into trajs        
        pindx = dcn 
        #print(f'For object id={idn}, nf = {nf}')
        for i in range(nf - 1): 
            # obtaining previous frame index
            pindx -= 1
            #print(len(detections), pindx)
            framedata = detections[pindx] 
            for obj in framedata: 
                if obj.id == idn:
                    # placing point into list of points 
                    x = obj.x
                    y = int(obj.y + obj.height / 2) 
                    pnt = [x, y]
                    pnts.append(pnt.copy())
                    break  # Break after finding the object
        # add id and list of points into dictionary          
        trajs.update({idn: pnts.copy()})
    #print('Exiting function obtainTrajs()')
    return trajs

def trajdiam(trajs, tkey): # измерить, сравнить время и результаты со старой версией
    ''' Computing diameter of the trajectory (maximum distance between points) 
        Input: trajs - dictionary with keys as number of trajectory, values as list of points
               tkey - number of the trajectory to compute diameter 
        Return: diameter of trajectory (float)
    ''' 
    # Check if there is a trajectory for a given tkey
    if tkey not in trajs or not trajs[tkey]:
        return 0.0 
    
    traj = np.array(trajs[tkey])  # Convert trajectory to numpy array
    npoints = len(traj)

    # If there are less than two points, the trajectory diameter is 0
    if npoints < 2:
        return 0.0
    
    # Compute pairwise differences
    diff = traj[:, np.newaxis, :] - traj[np.newaxis, :, :]
    
    # Compute pairwise distances
    distances = np.sqrt(np.sum(diff**2, axis=2))
    
    # Find the indices of the upper triangle (without diagonal)
    upper_triangle_indices = np.triu_indices(npoints, k=1)
    
    if len(upper_triangle_indices[0]) == 0:
        return 0.0  #If there are no pairs of points, return 0
    
    rmax = np.max(distances[upper_triangle_indices])
    return rmax

def trajdiam_old(trajs, tkey): # измерить, сравнить время и результаты с новой версией
    ''' Computing diameter of the trajectory (maximum distance between points) 
        Updated 29.11.2024 
        Input: trajs - dictionary with keys as number of trajectory, values as list of points
               tkey - number of the trajectory to compute diameter 
        Return: diameter of trajectory (float)
    ''' 
    traj = trajs[tkey]
    rmax = 0
    npoints = len(traj) 
    for i in range(0, npoints - 1, 1):
        for j in range(i, npoints, 1):    
            x1 = traj[i][0]
            y1 = traj[i][1]
            x2 = traj[j][0]
            y2 = traj[j][1]
            r = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if r > rmax:
                rmax = r
    return rmax

def validateObjs(detections: list[list[Detection]], frame_counter, fps, nf_threshold_id): 
    ''' Method to mark objects in the current framedata as suitable for reporting
        Criteria are (1) object appeared for over nf_threshold_id frames
                     (2) trajectory diameter is over tr_mov_threshold
                     (3) object was never reported (fnrlt = -1) 
                         or object was reported report_frame ago (n - int)   
        Updated 29.11.2024  
        Input: detections - circular list of lists of lists
               dcn - index of current framedata in detections
        Return: detections with property sfr set to True for objects satisfying the criteria 
        Global parameters: nf_threshold_id, tr_mov_threshold
    ''' 
    trajdiamTimeMes = []
    trajdiamOldTimeMes = []

    dcn = dcn = len(detections)-1
    nobsfr = 0
    for current_detection in detections[dcn]: 
        if current_detection.frames_count >= nf_threshold_id:
            idn = current_detection.id
            trajs = obtainTrajs(detections, dcn, [idn])

            start = time.time()
            movement = trajdiam(trajs, idn)
            trajdiamTimeMes.append(time.time()-start)

            start = time.time()
            movementTest = trajdiam_old(trajs, idn)
            trajdiamOldTimeMes.append(time.time()-start)

            if movement >= settings.tr_mov_threshold: 
                if current_detection.last_frame == -1:
                    current_detection.for_report = True
                    nobsfr += 1 
                elif current_detection.last_frame > -1 and (frame_counter - current_detection.last_frame)%(settings.report_frame) == 0:
                    current_detection.for_report = True
                    nobsfr += 1
    return detections, nobsfr, trajdiamTimeMes, trajdiamOldTimeMes

