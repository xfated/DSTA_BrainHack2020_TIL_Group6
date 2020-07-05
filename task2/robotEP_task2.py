from EP_api import findrobotIP, Robot
import cv2
import time
import threading
import argparse

from pid_control import *
from object_detection_models import * 

import colorsys
import numpy as np
from timeit import default_timer as timer
from PIL import Image, ImageFont, ImageDraw

from object_detection_models import get_clothes_class



## VARIABLES
robot_forward_check = False
desired_clothing = ['tops','trousers']
times_checked = 0
doll_found = False
reached_doll = False

fwd_check_dist = '1'
fwd_check_dist_back = '1'
fwd_time_sleep = 1
fwd_time_sleep_back = 1


robot_ip = findrobotIP()
robot = Robot(robot_ip)

'''
A simple matching algorithm
'''
def validate_detections(preds):
    global desired_clothing
    if len(desired_clothing) == 0:
        return 0
    else:
        print('validate_det')
        score = 0.0
        if len(preds)>0:  ##only do anything if there are any predictions at all
            item_list = []
            conf = []
            for pred in preds:
                item_list.append(categories[str(pred[1])])
                conf.append(pred[0])
                # E.g. ['dresses', 'outerwear']
            print(item_list)
            for item in item_list:
                if item in desired_clothing:
                    score += 1.0     # add one if the prediction matches that in nlp_list
            score = score/len(desired_clothing)  #calculate percentage of matching items. if >= 1, means predictions match
        return score

## code to execute during the 180 degree sweep for dolls
## return image_np with fashion
def sweeping_detection(preds, image_np):
    global robot_forward_check
    global times_checked
    print('sweeping_det')
    print('times checked: ', times_checked)
    # only do something if there are any predictions at all (above our threshold)
    if len(preds) > 0:
        xmid, ymid, area = get_center(preds[0][2])
        if xmid >= 0.35 and xmid <= 0.6:
            if robot_forward_check == False:
                robot._sendcommand('chassis move x {} vxy 0.3'.format(fwd_check_dist))
                time.sleep(fwd_time_sleep)
                robot._sendcommand('chassis wheel w1 0 w2 0 w3 0 w4 0')
                time.sleep(0.5)
                robot_forward_check = True
                return False, image_np
            
            else:
                ## only apply fashion model when robot has moved forward
                preds_fashion, image_np = show_inference(detection_model,image_np)
                print('preds_fashion: ', preds_fashion)
                score = validate_detections(preds_fashion)
                print('score: ',score)
                if score >= 1 :  #matches our target doll
                    ## turn green
                    robot._sendcommand('led control comp bottom_all r 0 g 255 b 0 effect scrolling')
                    time.sleep(3)  # halt to show "detection"
                    robot._sendcommand('led control comp bottom_all r 255 g 255 b 255 effect solid')
                    times_checked = 0
                    robot_forward_check = False
                    # found, move back
                    robot._sendcommand('chassis move x -{} vxy 0.3'.format(fwd_check_dist_back))
                    time.sleep(fwd_time_sleep_back)
                    robot._sendcommand('chassis wheel w1 0 w2 0 w3 0 w4 0')
                    time.sleep(0.5)
                    return True, image_np
                else:
                    ## turn red
                    if times_checked > 10:
                        robot._sendcommand('led control comp bottom_all r 255 g 0 b 0 effect scrolling')
                        time.sleep(3)  # halt to show "detection"
                        robot._sendcommand('led control comp bottom_all r 255 g 255 b 255 effect solid')    
                        times_checked = 0
                        robot_forward_check = False
                        robot._sendcommand('chassis move x -{} vxy 0.3'.format(fwd_check_dist_back))
                        time.sleep(fwd_time_sleep_back)
                        robot._sendcommand('chassis wheel w1 0 w2 0 w3 0 w4 0')
                        time.sleep(0.5)
                        return True, image_np

                    times_checked += 1
                    return False, image_np
        else:
            if robot_forward_check == True:
                ## only apply fashion model when robot has moved forward
                preds_fashion, image_np = show_inference(detection_model,image_np)
                print('preds_fashion: ', preds_fashion)
                score = validate_detections(preds_fashion)
                print('score: ',score)
                if score >= 1 :  #matches our target doll
                    ## turn green
                    robot._sendcommand('led control comp bottom_all r 0 g 255 b 0 effect scrolling')
                    time.sleep(3)  # halt to show "detection"
                    robot._sendcommand('led control comp bottom_all r 255 g 255 b 255 effect solid')
                    times_checked = 0
                    robot_forward_check = False
                    # found, move back
                    robot._sendcommand('chassis move x -{} vxy 0.3'.format(fwd_check_dist_back))
                    time.sleep(fwd_time_sleep_back)
                    robot._sendcommand('chassis wheel w1 0 w2 0 w3 0 w4 0')
                    time.sleep(0.5)
                    return True, image_np
                else:
                    ## turn red
                    if times_checked > 10:
                        robot._sendcommand('led control comp bottom_all r 255 g 0 b 0 effect scrolling')
                        time.sleep(3)  # halt to show "detection"
                        robot._sendcommand('led control comp bottom_all r 255 g 255 b 255 effect solid')    
                        times_checked = 0
                        robot_forward_check = False
                        robot._sendcommand('chassis move x -{} vxy 0.3'.format(fwd_check_dist_back))
                        time.sleep(fwd_time_sleep_back)
                        robot._sendcommand('chassis wheel w1 0 w2 0 w3 0 w4 0')
                        time.sleep(0.5)
                        return True, image_np

                    times_checked += 1
                    return False, image_np
            else:
                return False, image_np
    else:
        if robot_forward_check == True:
            if times_checked > 10:
                robot._sendcommand('led control comp bottom_all r 255 g 0 b 0 effect scrolling')
                time.sleep(3)  # halt to show "detection"
                robot._sendcommand('led control comp bottom_all r 255 g 255 b 255 effect solid')    
                times_checked = 0
                robot_forward_check = False
                robot._sendcommand('chassis move x -{} vxy 0.3'.format(fwd_check_dist_back))
                time.sleep(fwd_time_sleep_back)
                robot._sendcommand('chassis wheel w1 0 w2 0 w3 0 w4 0')
                time.sleep(0.5)
                return True, image_np

            times_checked += 1
            return False, image_np
        else:
            return False, image_np

def sweeping_detection_back(preds, image_np):
    print("GOING BACK")
    global robot_forward_check
    global times_checked
    global doll_found
    print('sweeping_det')
    print('times checked: ', times_checked)
    
    # only do something if there are any predictions at all (above our threshold)
    if len(preds) > 0:
        xmid, ymid, area = get_center(preds[0][2])
        if xmid >= 0.40 and xmid <= 0.65:
            if robot_forward_check == False:
                robot._sendcommand('chassis move x {} vxy 0.3'.format(fwd_check_dist))
                time.sleep(fwd_time_sleep)
                robot._sendcommand('chassis wheel w1 0 w2 0 w3 0 w4 0')
                time.sleep(0.5)
                robot_forward_check = True
                return False, image_np
            
            else:
                ## only apply fashion model when robot has moved forward
                preds_fashion, image_np = show_inference(detection_model, image_np)
                print('preds_fashion: ', preds_fashion)
                score = validate_detections(preds_fashion)
                print('score: ', score)
                if score >= 1 :  #matches our target doll
                    ## turn green
                    robot._sendcommand('led control comp bottom_all r 0 g 255 b 0 effect scrolling')
                    time.sleep(3)  # halt to show "detection"
                    robot._sendcommand('led control comp bottom_all r 255 g 255 b 255 effect solid')
                    times_checked = 0
                    robot_forward_check = False
                    doll_found = True
                    print("DOLL FOUND!")
                    robot.openarm()
                    time.sleep(2)
                    return True, image_np
                else:
                    ## turn red
                    if times_checked > 10:
                        robot._sendcommand('led control comp bottom_all r 255 g 0 b 0 effect scrolling')
                        time.sleep(3)  # halt to show "detection"
                        robot._sendcommand('led control comp bottom_all r 255 g 255 b 255 effect solid')    
                        times_checked = 0
                        robot_forward_check = False
                        robot._sendcommand('chassis move x -{} vxy 0.3'.format(fwd_check_dist_back))
                        time.sleep(fwd_time_sleep_back)
                        robot._sendcommand('chassis wheel w1 0 w2 0 w3 0 w4 0')
                        time.sleep(0.5)
                        return True, image_np

                    times_checked += 1
                    return False, image_np
        else:
            if robot_forward_check == True:
                ## only apply fashion model when robot has moved forward
                preds_fashion, image_np = show_inference(detection_model,image_np)
                print('preds_fashion: ', preds_fashion)
                score = validate_detections(preds_fashion)
                print('score: ',score)
                if score >= 1 :  #matches our target doll
                    ## turn green
                    robot._sendcommand('led control comp bottom_all r 0 g 255 b 0 effect scrolling')
                    time.sleep(3)  # halt to show "detection"
                    robot._sendcommand('led control comp bottom_all r 255 g 255 b 255 effect solid')
                    times_checked = 0
                    robot_forward_check = False
                    doll_found = True
                    print("DOLL FOUND!")
                    robot.openarm()
                    time.sleep(2)
                    return True, image_np
                else:
                    ## turn red
                    if times_checked > 10:
                        robot._sendcommand('led control comp bottom_all r 255 g 0 b 0 effect scrolling')
                        time.sleep(3)  # halt to show "detection"
                        robot._sendcommand('led control comp bottom_all r 255 g 255 b 255 effect solid')    
                        times_checked = 0
                        robot_forward_check = False
                        robot._sendcommand('chassis move x -{} vxy 0.3'.format(fwd_check_dist_back))
                        time.sleep(fwd_time_sleep_back)
                        robot._sendcommand('chassis wheel w1 0 w2 0 w3 0 w4 0')
                        time.sleep(0.5)
                        return True, image_np

                    times_checked += 1
                    return False, image_np
            else:
                return False, image_np
    else:
        if robot_forward_check == True:
            if times_checked > 10:
                robot._sendcommand('led control comp bottom_all r 255 g 0 b 0 effect scrolling')
                time.sleep(3)  # halt to show "detection"
                robot._sendcommand('led control comp bottom_all r 255 g 255 b 255 effect solid')    
                times_checked = 0
                robot_forward_check = False
                robot._sendcommand('chassis move x -{} vxy 0.3'.format(fwd_check_dist_back))
                time.sleep(fwd_time_sleep_back)
                robot._sendcommand('chassis wheel w1 0 w2 0 w3 0 w4 0')
                time.sleep(0.5)
                return True, image_np

            times_checked += 1
            return False, image_np
        return False, image_np
    
## code to execute when we want to retrieve doll
def retrieve_doll(preds, area_threshold = 0.5):
    # Center robot towards doll using top prediction
    global reached_doll
    if len(preds) > 0:
        xmid, ymid, area = get_center(preds[0][2])
        print(area)
        if (area < area_threshold):   #below area_threshold == far away. boxes don't occupy much of screen
            ## keep adjusting until is roughly center
            if not (xmid >= 0.45 and xmid <= 0.55):
                rotation_movement = pidCalculateRotation(setpoint = xmid) # how much to move rotation
                print("Robot is adjusting: ", rotation_movement)
                robot.rotate('{}'.format(rotation_movement))
            # Already ~ centered
            else: 
                #Move robot forward until reach 
                ## get total area of all bounding boxes
                print('moving to doll')
                # total_area = total_bbox_area(preds)
                robot._sendcommand('chassis move x 0.05 vxy 0.2')
                time.sleep(0.5)
        else:
            robot.stop()
            reached_doll = True
            print('near enough')
 
    else:
        print('Retrieving doll: no preds')

"""
Arguments
---------
bbox: 2nd argument of prediction. e.g. pred[2]
    ymin, xmin, ymax, xmax (relative to image width and size)

Returns: tuple
--------
xmid: mid x of bbox
ymid: mid y of bbox
area: area of bbox. Can use to gauge relative distance of object. Large --> near af
    note that is in terms of relative size. adjust assumptions accordingly
"""
def get_center(bbox):
    ymin, xmin, ymax, xmax = bbox
    xmid = (xmax+xmin)/2
    ymid = (ymax+ymin)/2
    area = (ymax-ymin) * (xmax-xmin)
    return xmid,ymid,area


def total_bbox_area(preds):
    total_area = 0.0
    for pred in preds:
        total_area += get_center(pred[2])[2]
    return total_area

''' Fill in your robotic response here '''
def move_robot():
	return None


def close_stream(robot):
    print("Quitting...")
    robot.exit()


def main():

    robot.startvideo()
    while robot.frame is None:
        pass
    #robot._sendcommand('camera exposure high')

    # Init Variables
    img_counter = 0
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    count = 0

    ## flags for control
    start_face_left = False
    sweeping_done = False
    degrees_rotated = 0
    doll_grabbed = False
    ########## Use this to prompt for an input mission ########## 
    txt = input("input mission statement: ")
    print('Mission statement', txt)

    extractor = get_clothes_class('./nlp_model.hdf5','./tokenizer.pkl','./encoded_words.pkl')
    desired_clothing = extractor.process_input(txt)
    print ('Extracted target clothing article: ',desired_clothing)
    
 

    ### Initialise the robot to run on autonomous mode
    mode = 'auto'
    # Lift arm up to elevate the camera angle
    robot._sendcommand('chassis move x 1.5 vxy 0.3')
    time.sleep(2)
    robot._sendcommand('chassis wheel w1 0 w2 0 w3 0 w4 0')
    time.sleep(0.5)
    robot._sendcommand('robotic_arm moveto x 210 y -40')
    time.sleep(1)
    robot.closearm()
    time.sleep(1)

    while True:

        cv2.namedWindow('Live video', cv2.WINDOW_NORMAL)
        cv2.imshow('Live video', robot.frame)

        # Key listener
        k = cv2.waitKey(16) & 0xFF


        ''' To provide toggle on keyboard between manual driving and autonomous driving '''
        if k == ord('m'):
            print('manual mode activated')
            mode = 'manual'

        if k == ord('n'):
            print('autonomous mode activated')
            mode = 'auto'

        if k == 27: # press esc to stop
            close_stream(robot)
            cv2.destroyWindow("result")
            cv2.destroyWindow("Live video")
            break

        frame = robot.frame
        frame_h, frame_w, frame_channels = frame.shape

        '''
        This set of codes is used for manual control of robot
        '''
        if mode == 'manual':            
            if k == ord('p'):
              robot.scan()
            
            if k == ord('w'):
              robot._sendcommand('chassis move x 0.2')

            elif k == ord('a'):
              robot._sendcommand('chassis move y -0.2')

            elif k == ord('s'):
              robot._sendcommand('chassis move x -0.2')

            elif k == ord('d'):
              robot._sendcommand('chassis move y 0.2')
            elif k == ord('q'):
              robot._sendcommand('chassis move z -5')
            elif k == ord('e'):
              robot._sendcommand('chassis move z 5')

            # elif k == ord('i'): # up and down arrow sometimes dont work
            #   robot._sendcommand('gimbal move p 1')
            # elif k == ord('k'): 
            #   robot._sendcommand('gimbal move p -1')
            # elif k == ord('j'): # up and down arrow sometimes dont work
            #   robot._sendcommand('gimbal move y -1')
            # elif k == ord('l'): 
            #   robot._sendcommand('gimbal move y 1')

            elif k == ord('r'): 
              robot._sendcommand('robotic_arm recenter')
            elif k == ord('x'): 
              robot._sendcommand('robotic_arm stop')
            elif k == ord('c'): 
              robot._sendcommand('robotic_arm moveto x 210 y 44')
            elif k == ord('z'): 
              robot._sendcommand('robotic_arm moveto x 92 y 90')
            elif k == ord('f'):     
              robot._sendcommand('robotic_gripper open 1')
            elif k == ord('g'): 
              robot._sendcommand('robotic_gripper close 1')

        elif mode=='auto':

            image_doll = frame.copy()
            image_np = frame.copy()
            '''
            Insert your model here 
            '''
            preds, image_np_dolls = show_inference_dolls(doll_model, image_doll)
            # print(preds)
            '''
            INSERT YOUR ROBOT RESPONSE MOVEMENT HERE
            '''
            ## try catch for any unexpected errors
            try:
                if sweeping_done == False:
                    if start_face_left == False:
                        robot.rotate('-90')
                        time.sleep(4)
                        start_face_left = True
                    else:
                        result, image_np = sweeping_detection(preds,image_np)
                        if result == False and times_checked == 0:
                            ## rotate 5 degree
                            robot.rotate('5')
                            time.sleep(0.3) 
                            degrees_rotated += 5
                        elif result == True:
                            robot.rotate('30')
                            time.sleep(2)
                            degrees_rotated += 30
                        else:
                            pass 
                        if degrees_rotated >= 220:
                            sweeping_done = True 
                # time for retrieval of doll
                elif reached_doll == False: 
                    # turn back to doll. rotate until match
                    if doll_found == False:
                        result, image_np = sweeping_detection_back(preds, image_np)
                        if result == False and times_checked == 0:
                            ## rotate 5 degree
                            robot.rotate('-5')
                            time.sleep(0.3) 
                            degrees_rotated += 5
                        elif result == True and doll_found == False:
                            robot.rotate('-30')
                            time.sleep(2)
                            degrees_rotated += 30
                        else:
                            pass  
                    # doll is found
                    else:
                        ## @TODO add logic for doll grabbed ==> stop robot
                        retrieve_doll(preds, area_threshold = 0.22)
                else:
                    if doll_grabbed == False:
                        print("reached doll, waiting for order")
                        # robot._sendcommand('chassis move x 0.05 vxy 0.1')
                        # robot._sendcommand('robotic_arm moveto x 270 y -40')
                        time.sleep(2)
                        robot.closearm()
                        time.sleep(2)
                        robot._sendcommand('robotic_arm moveto x 120 y 50')
                        print("Doll grabbed")
                        doll_grabbed = True
                    else:
                        robot._sendcommand('led control comp bottom_all r 0 g 0 b 255 effect solid')
                        time.sleep(1)
                        robot._sendcommand('led control comp bottom_all r 0 g 255 b 0 effect solid')
                        time.sleep(1)
                        robot._sendcommand('led control comp bottom_all r 255 g 0 b 0 effect solid')
                        time.sleep(1)

            except Exception as e:
                print(e)

            curr_time = timer()            
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            
            cv2.putText(image_np, text=fps, org=(3, 35), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.2, color=(0, 255, 0), thickness=2)
        
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", image_np)

if __name__ == '__main__':
    main()
