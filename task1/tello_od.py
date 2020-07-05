import cv2
import time
import threading
import argparse
import math

from od_letter_model import *

import colorsys
import numpy as np
from timeit import default_timer as timer
from PIL import Image, ImageFont, ImageDraw

filename = 'lollipop' 
IMG_PATH = "data/{}.jpg".format(filename) # name of image captured by tello
NEW_IMG_PATH = "data/{}_processed.jpg".format(filename) # for scoring


def close_stream(robot):
    print("Quitting...")
    robot.exit()

def getIfromRGB(rgb):
    red = rgb[0]
    green = rgb[1]
    blue = rgb[2]
    RGBint = (red<<16) + (green<<8) + blue
    return RGBint

def filterWhiteBoxes(maskedData):
    # image_np = cv2.imread('map.jpg')
    image_np = maskedData
    image_np = np.asarray(image_np)
    #height = no. of rows
    #width = no. of columns
    height, width = image_np.shape
#     print(height)
#     print(width)
    # f --> filter size == (f, f)
    # s --> stride = how many steps to jump for next filter
    s = 1
    threshold_4 = 10
    threshold_5 = 15
    left_found = False
    right_found = False
    output = []

    ## remove 4 if necessary. [4, 5]
    for f in [5, 4]:
        if right_found == True and left_found == True:
            continue #dont do 4by4 if 5by5 can find
        num_steps_w = int((width - f)/s) + 1 
        num_steps_h = int((height - f)/s) + 1

        for h in range(num_steps_h):
            h_start = h * s
            h_end = h_start + f
            
            for w in range(num_steps_w):
                w_start = w * s
                w_end = w_start + f
                # print('{} {} {} {}'.format(h_start, h_end, w_start, w_end))
                image_slice = image_np[h_start:h_end, w_start:w_end].astype(int)       
                if f == 5:
                    filter_box = np.array([
                        [1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1]
                    ])
                    sum = np.sum(np.sum((np.multiply(image_slice, filter_box))))
                    if sum >= threshold_5:
                        if (w_end+w_start)/2 < width/2:
                            left_found = True
                        else:
                            right_found = True
                        output.append([(h_end+h_start)/2, (w_end+w_start)/2])
                elif f == 4:
                    filter_box = np.array([
                        [1, 1, 1, 1],
                        [1, 0, 0, 1],
                        [1, 0, 0, 1],
                        [1, 1, 1, 1]
                    ])
                    sum = np.sum(np.sum((np.multiply(image_slice, filter_box))))
                    if sum >= threshold_4:
                        output.append([(h_end+h_start)/2, (w_end+w_start)/2])
                
    print(output)

    left_records = 0
    right_records = 0
    left_x_sum = 0
    left_y_sum = 0
    right_x_sum = 0
    right_y_sum = 0
    for i in range(len(output)):
        y = output[i][0]
        x = output[i][1]

        if x < width/2:
            left_x_sum += x
            left_y_sum += y
            left_records+=1
        else:
            right_x_sum += x
            right_y_sum += y
            right_records+=1
    
    left_records = 1 if left_records == 0 else left_records
    right_records = 1 if right_records == 0 else right_records
    print(right_records)
    left_x_sum = left_x_sum/left_records
    left_y_sum = left_y_sum/left_records
    right_x_sum = right_x_sum/right_records
    right_y_sum = right_y_sum/right_records
    
    left_box = [int(left_y_sum), int(left_x_sum)]
    right_box = [int(right_y_sum), int(right_x_sum)]
    print('left_box: ', left_box)
    print('right_box: ', right_box)
    return left_box, right_box

def get5x3Grid(maskedData, left_box, right_box):
    grid = [[0 for x in range(5)] for y in range(3)]         
    
    maskedData = np.array(maskedData).astype(int)
    threshold = 6

    # from left
    lr = left_box[0]
    lc = left_box[1]
    grid[1][1] = 10
    #[0][0]
    num = 0
    r = lr - 7
    c = lc - 7
    for i in range(5):
        for j in range(5):
            num += maskedData[r+i][c+j]
    num = 1 if num > threshold else 0  
    grid[0][0] = num
    
    #[0][1]
    num = 0
    r = lr - 7
    c = lc - 2
    for i in range(5):
        for j in range(5):
            num += maskedData[r+i][c+j]
    num = 1 if num > threshold else 0
    grid[0][1] = num
    
    #[0][2]
    num = 0
    r = lr - 7
    c = lc + 3
    for i in range(5):
        for j in range(5):
            num += maskedData[r+i][c+j]
    num = 1 if num > threshold else 0
    grid[0][2] = num
    
    #[1][0]
    num = 0
    r = lr - 2
    c = lc - 7
    for i in range(5):
        for j in range(5):
            num += maskedData[r+i][c+j]
    num = 1 if num > threshold else 0
    grid[1][0] = num
    
    #[1][2]
    num = 0
    r = lr - 2
    c = lc + 3
    for i in range(5):
        for j in range(5):
            num += maskedData[r+i][c+j]
    num = 1 if num > threshold else 0
    grid[1][2] = num
    
    #[2][0]
    num = 0
    r = lr + 3
    c = lc - 7
    for i in range(5):
        for j in range(5):
            num += maskedData[r+i][c+j]
    num = 1 if num > threshold else 0
    grid[2][0] = num
    
    #[2][1]
    num = 0
    r = lr + 3
    c = lc - 2
    for i in range(5):
        for j in range(5):
            num += maskedData[r+i][c+j]
    num = 1 if num > threshold else 0
    grid[2][1] = num
    
    #[2][2]
    num = 0
    r = lr - 7
    c = lc + 3
    for i in range(5):
        for j in range(5):
            num += maskedData[r+i][c+j]
    num = 1 if num > threshold else 0
    grid[2][2] = num
    
    # from right
    lr = right_box[0]
    lc = right_box[1]
    grid[1][3] = 10
    #[0][3]
    num = 0
    r = lr - 7
    c = lc - 2
    for i in range(5):
        for j in range(5):
            num += maskedData[r+i][c+j]
    num = 1 if num > threshold else 0
    grid[0][3] = num
    
    #[0][4]
    num = 0
    r = lr - 7
    c = lc + 3
    for i in range(5):
        for j in range(5):
            num += maskedData[r+i][c+j]
    num = 1 if num > threshold else 0
    grid[0][4] = num
    
    #[1][4]
    num = 0
    r = lr - 2
    c = lc + 3
    for i in range(5):
        for j in range(5):
            num += maskedData[r+i][c+j]
    num = 1 if num > threshold else 0
    grid[1][4] = num
    
    #[2][3]
    num = 0
    r = lr + 3
    c = lc - 2
    for i in range(5):
        for j in range(5):
            num += maskedData[r+i][c+j]
    num = 1 if num > threshold else 0
    grid[2][3] = num
    
    #[2][4]
    num = 0
    r = lr + 3
    c = lc + 3
    for i in range(5):
        for j in range(5):
            num += maskedData[r+i][c+j]
    num = 1 if num > threshold else 0
    grid[2][4] = num
    
    grid.append([1, 1, 1, 1, 1])
    grid.reverse()
    grid.append([1, 1, 1, 1, 1])
    grid.reverse()
    
    return grid

def applyBbox(bbox, height, width):
    ymin = bbox[0] * height
    xmin = bbox[1] * width
    ymax = bbox[2] * height
    xmax = bbox[3] * width
    pos = "e" if ymin < height/2 else "s"
    y_pos = 0 if pos == "e" else 4
    
    x_center = (xmin+xmax)/2
    print(x_center)
    
    width_size = width // 3
    parts = []

    print("width: ",width)
    print("widthsize: ", width_size)
    for i in range(0, width-width_size+100, width_size):
        part = [i, i + width_size]
        parts.append(part)
    print('parts: ', parts)
    
    for i in range(3): 
        if parts[i][0] < x_center < parts[i][1]:
            x_pos = i
            break


    if x_pos == 0:
        x_pos = 0
    elif x_pos == 1:
        x_pos = 2
    else:
        x_pos = 4
   
    return pos, (y_pos, x_pos)

def BFS(board, start):
  queue = list()
  queue.append(start)
  visited = set()

  # this keeps track of where did we get to each vertex from
  # so that after we find the exit we can get back
  parents = dict()
  parents[start] = None

  while queue:
    v = queue.pop(0)
    if board[v[0]][v[1]] == 'E':
      break
#     queue = queue[1:]   # this is inefficient, an actual queue should be used 
    visited.add(v)
    for u in neighbors(board, v):
      if u not in visited:
        parents[u] = v
        queue.append(u)
    
#     print(queue)

  # we found the exit, now we have to go through the parents 
  # up to the start vertex to return the path
  path = list()
  while v != None:
    path.append(v)
#     print(path)
    v = parents[v]

  # the path is in the reversed order so we reverse it 
  path.reverse()
  return path

def neighbors(board, v):
  # right, left, up, down
  diff = [(0, 1), (0, -1), (1, 0), (-1, 0)]
  retval = list()
  for d in diff:
    newr = d[0] + v[0]
    newc = d[1] + v[1]
    if newr < 0 or newr >= len(board) or newc < 0 or newc >= len(board[0]):
      continue
    if board[newr][newc] == 0 or board[newr][newc] == 'E':  
        retval.append((newr, newc))
        
  return retval

def getDirections(path):
    directions = list()
    path.pop(0) # remove first direction for robot
    
    node1 = path.pop(0)
    while path:
        node2 = path.pop(0)
        if node1[0] == node2[0]: # same row
            if node1[1] > node2[1]:
                directions.append("left")
            else:
                directions.append("right")
        elif node1[1] == node2[1]: # same col
            if node1[0] > node2[0]:
                directions.append("front")
            else:
                directions.append("down")
        node1 = node2
    
    directions.pop() # remove last direction to change to "end" later
    
    robotInstructions = list()
    while directions:
        d = directions.pop(0)
        directions.pop(0)
        robotInstructions.append(d)
    robotInstructions.append("end")
    
    return robotInstructions


def get_image():
    global IMG_PATH
    global NEW_IMG_PATH
    img = Image.open(IMG_PATH)
    img_with_line = img.copy()

    img.getdata()
    image_np = np.array(img)
    image_np = image_np[:, :, ::-1].copy() 
    width, height = img.size
    print(width, height)
    boxes = {'s': [], 'e': []} 
    # topleft: [0.        , 0.11049234, 0.12848783, 0.18082973] topmid: [0.04847554, 0.43198667, 0.14931671, 0.4866951 ] topright: [0.04847554, 0.83198667, 0.14931671, 0.8866951 ]
    # btmmid: [0.7444849 , 0.48952463, 0.8416772 , 0.5327652 ] btmright: [0.85544849, 0.83198667, 0.8416772, 0.9066951 ]
    print('empty boxes: ', boxes)
    preds, image_se_boxes = show_inference(detection_model, image_np)
    s_conf = 0
    e_conf = 0
    for pred in preds:
        if pred[1] == 1: #start
            if pred[0] > s_conf:
                s_conf = pred[0]
                boxes['s'] = pred[2]
    for pred in preds:
        if pred[1] == 2: #start
            if pred[0] > e_conf:
                e_conf = pred[0]
                boxes['e'] = pred[2]
    
    print('boxes: ', boxes)
    new_width  = 32
    new_height = 24

    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    WIDTH, HEIGHT = img.size
    data = list(img.getdata())  

    data = [getIfromRGB(row) for row in data]
    OldMax = max(data)
    OldMin = min(data)
    NewMax = 255
    NewMin = 0

    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    dataz = []

    for row in data:
        dataz.append((((row - OldMin) * NewRange) / OldRange) + NewMin)

    dataz = [dataz[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]

    chars = '0123456789'  # Change as desired.
    scale = (len(chars)-1)/255.

    dataz = [[int(chars[int(pix*scale)]) for pix in row] for row in dataz]

    dataz = [[0 if pix < 5 else 1 for pix in row] for row in dataz]
    print(dataz)
    # deepcopy dataz
    maskedData = [[0 for x in range(32)] for y in range(24)] 
    for row in range(0, 24):
        for col in range(0, 32):
            maskedData[row][col] = dataz[row][col]
    print('Before Mask:')
    for row in maskedData:
        print(' '.join(str(value) for value in row)) 
        
    #top
    for i in range(0,6):
        for j in range(0,32):
            maskedData[i][j] = "0"
            
    #bottom
    for i in range(19,24):
        for j in range(0,32):
            maskedData[i][j] = "0"
            
    for i in range(0,23):
        for j in range(0,6):
            maskedData[i][j] = "0"
            
    for i in range(0,23):
        for j in range(27,32): # ride side mask 4 columns (just for this image)
            maskedData[i][j] = "0"

    for i in range(0,23):
        for j in range(15,16):
            maskedData[i][j] = "0"

    print("Masked:")
    for row in maskedData:
        print(' '.join(str(value) for value in row)) 

    one, two = filterWhiteBoxes(maskedData)   
    x_dist = two[1] - one[1]
    grid_dist = (x_dist/new_width)*width
    print('grid_dist: ', grid_dist)
    # Uncomment if necessary
    # apply mask for simulated crop if using official picture
    # crop_pad = 3
    # maskedData = [[0 for x in range(32)] for y in range(24)] 
    # for row in range(0, 24):
    #     for col in range(0, 32):
    #         maskedData[row][col] = dataz[row][col]
    # for i in range(0,crop_pad):
    #     for j in range(0,32):
    #         maskedData[i][j] = 0
            
    # for i in range(24 - crop_pad,24):
    #     for j in range(0,32):
    #         maskedData[i][j] = 0
            
    # for i in range(0,23):
    #     for j in range(0,crop_pad):
    #         maskedData[i][j] = 0
            
    # for i in range(0,23):
    #     for j in range(32-crop_pad,32): # ride side mask 4 columns (just for this image)
    #         maskedData[i][j] = 0

    # print("Crop mask:")
    # for row in maskedData:
    #     print(' '.join(str(value) for value in row)) 

    grid = get5x3Grid(dataz, one, two)  #used maskedData (masked) / dataz (no mask)

    pos1, c1 = applyBbox(boxes['e'], height, width)
    pos2, c2 = applyBbox(boxes['s'], height, width)
    # pos1, c1 = applyBbox([0.81925493, 0.47620693, 0.9512724,  0.5271045], height, width)
    # pos2, c2 = applyBbox([1.7291705e-04, 7.5504309e-01, 9.0263382e-02, 8.0393028e-01], height, width)
    print(c1)
    print(c2)
    start = (0,0)
    end = (0,0)
    if pos1 == 's':
        start = c1
        end = c2
    elif pos2 == 's':
        start = c2
        end = c1
    print(start)
    print(end)
    grid[start[0]][start[1]] = 'S'
    grid[end[0]][end[1]] = 'E'

    for row in grid:
        print(' '.join(str(value) for value in row)) 

    output = BFS(grid, start)
    print(output)

    directions = getDirections(output)
    sx_mid, sy_mid, _ = get_center(boxes['s'])
    sx_mid = sx_mid * width
    sy_mid = sy_mid * height 
    draw = ImageDraw.Draw(img_with_line)
    ## draw start
    cur_x = sx_mid
    cur_y = sy_mid
    next_x = sx_mid
    next_y = sy_mid - grid_dist/3
    draw.line((cur_x, cur_y, next_x, next_y), fill = (255,182,193,100), width = 10)
    for direction in directions:
        print('drawing')
        cur_x = next_x
        cur_y = next_y 
        if direction == 'left':
            next_x = cur_x - grid_dist
            next_y = cur_y
        elif direction == 'right':
            next_x = cur_x + grid_dist
            next_y = cur_y
        elif direction == 'front':
            next_x = cur_x
            next_y = cur_y - grid_dist
        elif direction == 'down':
            next_x = cur_x
            next_y = cur_y + grid_dist
        elif direction == 'end':    
            next_x = cur_x 
            next_y = cur_y - grid_dist/3
        
        draw.line((cur_x, cur_y, next_x, next_y), fill = (255,182,193,100), width = 10)
    img_with_line.save(NEW_IMG_PATH)
    print("instructions:", directions)
    return directions
