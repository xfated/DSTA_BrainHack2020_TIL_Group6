from EP_api import Robot, findrobotIP
import time
import numpy as np

robot = Robot(findrobotIP()) # router

def neeh():
    # robot._sendcommand('robotic_gripper open')
    # time.sleep(0.3)
    # robot._sendcommand('chassis move x 0.2 vxy 0.2')
    # time.sleep(1)
    robot._sendcommand('robotic_gripper open')
    time.sleep(0.4)
    robot._sendcommand('robotic_gripper close')
    time.sleep(0.3)
    robot._sendcommand('robotic_gripper open')
    time.sleep(0.3)
    robot._sendcommand('robotic_gripper close')
    time.sleep(0.3)
    # robot._sendcommand('robotic_gripper open')
    # time.sleep(0.3)
    # robot._sendcommand('robotic_gripper close')
    # time.sleep(0.3)
    


def funeral_dance():
    robot._sendcommand('robotic_arm moveto x 50 y 200')
    time.sleep(1)
 
    # turn right and slide forward
    robot._sendcommand('chassis move z 45 vz 100')
    time.sleep(0.8)
    robot._sendcommand('chassis move x 0.3 y -0.3 vxy 2')
    time.sleep(1)

    # # turn left and slide forward
    robot._sendcommand('chassis move z -90 vz 100')
    time.sleep(0.8)
    robot._sendcommand('chassis move x 0.3 y 0.3 vxy 2')
    time.sleep(1)
    
    # # turn back face forward
    robot._sendcommand('chassis move z 45 vz 100')
    time.sleep(0.8)

    # # slide left right then back to center
    robot._sendcommand('chassis move y -0.3 vxy 1.5')
    time.sleep(1)
    robot._sendcommand('chassis move y 0.6 vxy 1.5')
    time.sleep(1)
    robot._sendcommand('chassis move y -0.3 vxy 1.5')
    time.sleep(1)

    # # turn right and slide backward
    robot._sendcommand('chassis move z -45 vz 100')
    time.sleep(0.8)
    robot._sendcommand('chassis move x -0.3 y -0.3 vxy 2')
    time.sleep(0.8)

    # # turn left and slide backward
    robot._sendcommand('chassis move z 90 vz 100')
    time.sleep(0.8)
    robot._sendcommand('chassis move x -0.3 y 0.3 vxy 2')
    time.sleep(0.8)
    robot._sendcommand('chassis move z -45 vz 100')
    time.sleep(0.8)

    robot._sendcommand('chassis move x 0.3 vxy 2')
    time.sleep(0.8)
    
    # spinning
    robot._sendcommand('chassis move z 360 vz 200')
    time.sleep(2.5)
    robot._sendcommand('chassis move z -360 vz 200')
    time.sleep(2.5)

    # lights
    robot._sendcommand('led control comp bottom_all r 255 g 0 b 0 effect scrolling')

    robot._sendcommand('chassis move x 0.2 vxy 2')
    time.sleep(2)
    robot._sendcommand('robotic_gripper open')
    time.sleep(1)
    robot._sendcommand('robotic_arm moveto x 200 y -40')
    time.sleep(1)
    # robot._sendcommand('chassis move x 0.1 vz 100')
    # time.sleep(1)
    robot._sendcommand('robotic_gripper close')
    time.sleep(1.5)
    robot._sendcommand('robotic_arm moveto x 80 y 120')
    time.sleep(1)

    robot._sendcommand('led control comp bottom_all r 0 g 255 b 0 effect scrolling')
    
    # spin with mozart
    robot._sendcommand('chassis move z 360 vz 150')
    time.sleep(2.5)
    robot._sendcommand('robotic_arm moveto x 250 y 160')
    time.sleep(1)
    robot._sendcommand('chassis move z -360 vz 150')
    time.sleep(2.5)

    robot._sendcommand('robotic_arm moveto x 50 y 200')
    time.sleep(1)
    robot._sendcommand('robotic_arm moveto x 80 y 0')
    time.sleep(1)
    
    robot._sendcommand('robotic_arm moveto x 50 y 200')
    time.sleep(0.5)
    robot._sendcommand('robotic_arm moveto x 80 y 0')
    time.sleep(0.5)
    robot._sendcommand('robotic_arm moveto x 50 y 200')
    time.sleep(0.5)

    # # turn right and slide backward
    robot._sendcommand('chassis move z -45 vz 100')
    time.sleep(0.8)
    robot._sendcommand('chassis move x -0.3 y -0.3 vxy 2')
    time.sleep(0.8)

    # # turn left and slide backward
    robot._sendcommand('chassis move z 90 vz 100')
    time.sleep(0.8)
    robot._sendcommand('chassis move x -0.3 y 0.3 vxy 2')
    time.sleep(0.8)
    robot._sendcommand('chassis move z -45 vz 100')
    time.sleep(0.8)

    robot._sendcommand('chassis move z -90 vz 100')
    time.sleep(0.8)
    robot._sendcommand('chassis move x 0.3 vxy 2')
    time.sleep(0.8)
    robot._sendcommand('robotic_arm moveto x 200 y -40')
    time.sleep(1)
    robot._sendcommand('robotic_gripper open')
    time.sleep(1)
    robot._sendcommand('robotic_arm moveto x 80 y 120')
    time.sleep(1)
    robot._sendcommand('chassis move x -0.3 vxy 2')
    time.sleep(0.8)
    robot._sendcommand('chassis move z -90 vz 100')
    time.sleep(0.8)

    robot._sendcommand('led control comp bottom_all r 255 g 0 b 0 effect scrolling')
    robot._sendcommand('chassis move x -0.5 vxy 2')
    time.sleep(0.8)
    robot._sendcommand('chassis move z -20 vz 150')
    time.sleep(0.2)
    robot._sendcommand('chassis move z 40 vz 150')
    time.sleep(0.2)
    robot._sendcommand('chassis move z -40 vz 150')
    time.sleep(0.2)
    robot._sendcommand('chassis move z 40 vz 150')
    time.sleep(0.2)
    robot._sendcommand('chassis move z -40 vz 150')
    time.sleep(0.2)
    robot._sendcommand('chassis move z 40 vz 150')
    time.sleep(0.2)
    robot._sendcommand('chassis move z -40 vz 150')
    time.sleep(0.2)
    robot._sendcommand('chassis move z 40 vz 150')
    time.sleep(0.2)
    robot._sendcommand('chassis move z -40 vz 150')
    time.sleep(0.2)
    robot._sendcommand('chassis move z 40 vz 150')
    time.sleep(0.2)
    robot._sendcommand('chassis move z -40 vz 150')
    time.sleep(0.2)
    robot._sendcommand('chassis move z 40 vz 150')
    time.sleep(0.2)
    robot._sendcommand('chassis move z -40 vz 150')
    time.sleep(0.2)
    
       
    

if __name__ == "__main__":
    # neeh()
    funeral_dance()
    # dance_iter1()