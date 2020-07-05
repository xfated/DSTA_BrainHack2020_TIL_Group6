from EP_api import Robot, findrobotIP
import time
import cv2
import numpy as np
from Tello_api import Tello
from tello_od import *
robot = Robot(findrobotIP()) # router

## to be set by tello 
instructions = []

# 1: Tello move and capture image. Manually adjust Tello with keyboard. Press spacebar to capture and save image, press enter to retrieve Tello
tello = Tello()

tello.startvideo() 
tello.startstates()

while tello.frame is None:
	pass

tello._sendcommand('takeoff')
tello.start_pad_det()
padreply = tello._sendcommand('jump 100 0 120 60 0 m1 m2') # x y z speed yaw pad_ids
if padreply == "ok":
		print("Reached m2")

while True:
	cv2.namedWindow('Tello video', cv2.WINDOW_NORMAL)
	cv2.imshow('Tello video', cv2.flip(tello.frame,0))

	k = cv2.waitKey(16) & 0xFF
	if k == 32: # spacebar to capture image 
		cv2.imwrite(IMG_PATH, cv2.flip(tello.frame,0))
		print(f'Saved as {IMG_PATH}!')
		cv2.imshow('Saved Image', cv2.flip(tello.frame,0)) 

	elif k == 13: # enter to retrieve drone, land and continue
		padreply = tello._sendcommand('jump -100 0 100 60 0 m2 m1') # x y z speed yaw pad_ids
		if padreply == "ok":
				print("Reached m1")
		print('Landing...')
		tello.exit()
		break

	elif k != -1:  # press wasdqe to adjust position of drone, uj to adjust height
		tello.act(k) 
cv2.destroyAllWindows()



####
if __name__ == "__main__":
    instructions = get_image()


    ### ROBOT
    num_instructions_left = len(instructions)

    # Sides
    side_distance = 0.8
    side_sleep = 4

    # Forward
    forward_distance = 0.8
    forward_sleep = 3.15

    # Back
    back_distance = 0.8
    back_sleep = 4

    # Start/End
    start_distance = 0.52#0.55
    end_distance = 0.5#0.5
    start_end_sleep = 3

    robot.startvideo()
    while robot.frame is None:
        pass

    height, width, _ = robot.frame.shape
    print('height: ', height)
    print('width: ', width)
    mode = 'auto'
    # Lift arm up to elevate the camera angle
    robot._sendcommand('robotic_arm moveto x 74 y 34')
    time.sleep(1)

    # Key listener

    while True:
        cv2.namedWindow('Live video', cv2.WINDOW_NORMAL)
        
        image = robot.frame
       
        cv2.imshow('Live video', image)
        
        k = cv2.waitKey(16) & 0xFF

        if k ==27: #press esc to stop
            robot.exit()
            break

            ### Initialise the robot to run on autonomous mode
        

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
            # elif k == ord('j'): # up eand down arrow sometimes dont work
            #   robot._sendcommand('gimbal move y -1')
            # elif k == ord('l'): 
            #   robot._sendcommand('gimbal move y 1')
            elif k == ord('i'):
                robot._sendcommand('robotic_arm move x 0 y 50')
            elif k == ord('j'):
                robot._sendcommand('robotic_arm move x 50 y 0')
            elif k == ord('k'):
                robot._sendcommand('robotic_arm move x 0 y -50')
            elif k == ord('l'):
                robot._sendcommand('robotic_arm move x -50 y 0')

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

            # print(robot._sendcommand('robotic_arm position ?'))

        elif mode=='auto':

            # Move robot to the middle of first tile

            # Move bit by bit
            # robot._sendcommand('chassis move x 0.55')
            robot._sendcommand('chassis move x {} vxy 0.3'.format(start_distance)) 
            time.sleep(start_end_sleep)
            robot._sendcommand('chassis wheel w1 0 w2 0 w3 0 w4 0')
            time.sleep(0.2)


            while len(instructions) > 0: 

                instruction = instructions[0]

                if instruction == 'left':
                    robot._sendcommand('chassis move y -{} vxy 0.3'.format(side_distance)) 
                    time.sleep(side_sleep)
                    robot._sendcommand('chassis wheel w1 0 w2 0 w3 0 w4 0')
                    time.sleep(0.2)
                    instructions.pop(0)  

                elif instruction == 'right':     
                    robot._sendcommand('chassis move y {} vxy 0.3'.format(side_distance)) 
                    time.sleep(side_sleep)
                    robot._sendcommand('chassis wheel w1 0 w2 0 w3 0 w4 0')
                    time.sleep(0.2)
                    instructions.pop(0)

                elif instruction == 'front':
                    robot._sendcommand('chassis move x {} vxy 0.3'.format(forward_distance))  
                    time.sleep(forward_sleep)
                    robot._sendcommand('chassis wheel w1 0 w2 0 w3 0 w4 0')
                    time.sleep(0.2)
                    instructions.pop(0) 
                
                elif instruction == 'down':
                    robot._sendcommand('chassis move x -{} vxy 0.3'.format(back_distance))  
                    time.sleep(back_sleep)
                    robot._sendcommand('chassis wheel w1 0 w2 0 w3 0 w4 0')
                    time.sleep(0.2)
                    instructions.pop(0) 
                
                elif instruction == 'end':
                    robot._sendcommand('chassis move x {} vxy 0.3'.format(end_distance)) 
                    time.sleep(start_end_sleep)
                    robot._sendcommand('chassis wheel w1 0 w2 0 w3 0 w4 0')
                    time.sleep(0.2)
                    instructions.pop(0)
                  
            if len(instructions) == 0:
                robot._sendcommand('led control comp bottom_all r 0 g 0 b 255 effect solid')
                time.sleep(1)
                robot._sendcommand('led control comp bottom_all r 0 g 255 b 0 effect solid')
                time.sleep(1)
                robot._sendcommand('led control comp bottom_all r 255 g 0 b 0 effect solid')
                time.sleep(1)
                robot.exit()
        
                    
            # if we fail to reach, just give instructions to keep left all the way until we reach lol   
             
