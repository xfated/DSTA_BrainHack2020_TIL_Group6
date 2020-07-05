import cv2
import numpy as np


def canny(image):
    """ Obtain edges """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_img = cv2.Canny(gray, 50, 150)
    return canny_img

"""
@TODO Get mask region based on robot view
"""
def region_of_interest(image):
    """ Apply mask in other areas edges """
    height, width = image.shape
    ## Declare regions of interest
    polygons = np.array([
        [(0, height/2), (0, height), (480, 720), (480, 400), (800, 400), (800, 720), (width, height), (width, height/2)]  # points to declare region
    ], dtype = np.int32)
    ## create canvas
    mask = np.zeros_like(image)
    ## fill regions of interest with white
    cv2.fillPoly(mask, polygons, 255)
    ## apply mask on original image
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    ## create canvas
    line_image = np.zeros_like(image)
    ## only run if there are any lines detected
    if lines is not None:  
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line #.reshape(4)
                ## draw lines on line_image. (start point), (end point), (color), thickness
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # determine threshold for horz lines
            # x1, y1, x2, y2 = line.reshape(4)
            # slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
            # print(slope)
            # if abs(slope) < 0.3:
            #     cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
    
    return line_image

def analyze_lines(lines, image_shape):
    height, width, _ = image_shape
    left_lines = []
    right_lines = []
    horz_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            slope, intercept = np.polyfit((x1, x2), (y1, y2), 1) ## obtain parameters for polynomial of degree 1 (line)  
            if abs(slope) < 0.3:
                horz_lines.append((x1, y1, x2, y2))
            elif slope < 0:
                left_lines.append((x1, y1, x2, y2))
            else:
                right_lines.append((x1, y1, x2, y2))
    print('left lines: ',left_lines)
    print('right lines: ', right_lines)
    print('horz lines: ', horz_lines)


"""
@TODO Set boundary for nearest horizontal line
      Robot makes decision when nearest_left = None, nearest_right = None
"""
def get_closest_lines(lines, image_shape):
    height, width, _ = image_shape
    left_nearest_x = 0
    nearest_left = None
    right_nearest_x = width
    nearest_right = None
    horz_max_y = 0
    nearest_horz = None

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            slope, intercept = np.polyfit((x1, x2), (y1, y2), 1) ## obtain parameters for polynomial of degree 1 (line)  
            if abs(slope) < 0.3:
                ## lowest horizontal line will have greatest x value
                horz_y = max(y1, y2) # higher y value
                if horz_y < height*0.3: # top half of images 
                    if horz_y > horz_max_y:
                        horz_max_y = horz_y
                        nearest_horz = (x1, y1, x2, y2)
            elif slope < 0:
                # left line
                lowest_y = max(y1, y2)
                nearest_x = max(x1, x2) 
                if lowest_y > height*0.5:  # bottom half
                    if nearest_x < width*0.5:  # left side
                        if nearest_x > left_nearest_x: # corner is more right
                            left_nearest_x = nearest_x
                            nearest_left = (x1, y1, x2, y2)
            else:
                ## nearest right light, positive gradient. closest one will have smallest magnitude gradient. 
                lowest_y = max(y1, y2)
                nearest_x = min(x1, x2)
                if lowest_y > height*0.5:  # bottom half
                    if nearest_x > width*0.5:  #right side
                        if nearest_x < right_nearest_x: # corner is more left
                            right_nearest_x = nearest_x
                            nearest_right = (x1, y1, x2, y2)
    return (nearest_left, nearest_right, nearest_horz)

def calculate_line_mid(line):
    x1, y1, x2, y2 = line
    mid_x = (x1 + x2)/2
    mid_y = (y1 + y2)/2
    return mid_x, mid_y

if __name__ == "__main__":
    image = cv2.imread('./task1/side_view.jpg') # open image
    lane_image = np.copy(image)          # create copy
    canny_img = canny(lane_image)        # obtain edges
    cropped_image = region_of_interest(canny_img)   # apply mask
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi/100, threshold=100, lines=np.array([]), minLineLength=40, maxLineGap = 5)  # obtain lines
    #analyze_lines(lines, image.shape)
    lines = get_closest_lines(lines, image.shape)
    line_image = display_lines(lane_image, lines)   # show lines
    overlay_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)  # combine with original image
    cv2.imshow("result", overlay_image)           
    cv2.waitKey(0)

    """ For robot 
    from EP_api import Robot, findrobotIP
    import time

    robot = Robot(findrobotIP()) # router
    
    instructions = ['left']
    num_instructions_left = len(instructions)
    
    robot.startvideo()
    while robot.frame is None:
        pass

    height, width, _ = robot.frame.shape

    while True:
        cv2.namedWindow('Live video', cv2.WINDOW_NORMAL)
        
        image = robot.frame
        canny_img = canny(lane_image)        # obtain edges
        cropped_image = region_of_interest(canny_img)   # apply mask
        lines = cv2.HoughLinesP(canny_img, rho=2, theta=np.pi/100, threshold=100, lines=np.array([]), minLineLength=40, maxLineGap = 5)  # obtain lines
        closest_lines = get_closest_lines(lines, image.shape)

        # Display closest lines for reference
        line_image = display_lines(lane_image, closest_lines)   # show lines
        overlay_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)  # combine with original image

        cv2.imshow('Live video', overlay_image)

        k = cv2.waitKey(16) & 0xFF
        if k ==27: #press esc to stop
            robot.exit()
            break

        # Move robot
        nearest_left, nearest_right, nearest_horz = closest_lines
        if num_instructions_left > 0:
                
            # Move bit by bit
            robot.forward('0.2')

            # Adjust robot
            if nearest_left is not None:
                nearest_left_x, nearest_left_y = calculate_line_mid(nearest_left)
                if nearest_left_x > width*0.2:  # if exceed left boundary. i.e too near
                    robot.rotate('1') #turn right
                    # robot slide right
            if nearest_right is not None:
                nearest_right_x, nearest_right_y = calculate_line_mid(nearest_right)
                if nearest_right_x < width*0.8:  # if exceed right boundary. i.e too near
                    robot.rotate('-1') #turn left

            # cross junction || end of road. make decision to turn
            if nearest_left is None and nearest_right is None or nearest_horz is not None:
            # left / right / stop (if is at end)
                ###
                # make robot move given 
                ###
                instruction = instructions[len(instructions) - num_instructions_left] 
                if instruction == 'left':
                    robot.rotate('=90')
                elif instruction == 'right':
                    robot.rotate('90')

                num_instructions_left -= 1 # after execution
                
                ## get out of the cross junction
                robot.forward('0.5')
                time.sleep(1)
                
        # if we fail to reach, just give instructions to keep left all the way until we reach lol    

    """
