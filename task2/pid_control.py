pre_error_rotation = 0
pre_error_pitch = 0
rotation_integral = 0
pitch_integral = 0

# Rotation coefficients (degrees)
KP_rotation = 40   #proportional coefficient
KD_rotation = 0.00   #differential coefficient
KI_rotation = 0   #integral coefficient

# Pitch coefficients
KP_pitch = 1
KD_pitch = 0.05
KI_pitch = 0

""" 
Arguments
---------
current_val: float
    current value that signifies where our arm is at.
    (if is based on camera, might set to always be center of the screen == 0.5)
setpoint: float
    where we want the arm to go to. i.e. xmid, ymid of bounding box perhaps
Returns float
-------
    How much we need to move the arm (or whatever) by
-------
"""
def pidCalculateRotation(setpoint, current_val = 0.5):
    global rotation_integral
    global pre_error_rotation
    error = setpoint - current_val
    rotation_integral += error
    
    # calculate output
    Pout = KP_rotation * error
    Dout = KD_rotation * (error - pre_error_rotation)
    Iout = KI_rotation * rotation_integral
    pre_error_rotation = error
    
    return Pout + Dout + Iout

# def pidCalculatePitch(setpoint, current_val = 0.5):
#     error = setpoint - current_val
#     pitch_integral += error
    
#     # calculate output
#     Pout = KP_pitch * error
#     Dout = KD_pitch * (error - pre_error_pitch)
#     Iout = KI_pitch * integral
#     pre_error_pitch = error
    
#     return Pout + Dout + Iout