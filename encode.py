import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import sys

def step1(): # Convert RGB to YCbCr
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    YCrCb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y_img, Cr_img, Cb_img = cv2.split(YCrCb_img)

    plt.imshow(RGB_img)
    plt.imshow(Y_img, cmap='gray')
    plt.imshow(Cr_img, cmap='gray')
    plt.imshow(Cb_img, cmap='gray')
    return Y_img, Cr_img, Cb_img

def step2(Y_img): # Get eye macroblock
    Y_numarray = Y_img.astype('float')
    Y_numarray = pd.DataFrame(Y_numarray)
    
    plt.imshow(Y_numarray, cmap='gray')

    # Coordinate of the border of black and white eyes = (605, 488)
    x_coord = 616//8 * 8
    y_coord = 495//8 * 8

    makroblock = Y_numarray.iloc[y_coord:y_coord+8, x_coord:x_coord+8]
    makroblock = makroblock.to_numpy()
    print(makroblock)

    pd.DataFrame(makroblock)
    plt.imshow(makroblock, cmap='gray')

    return makroblock

def step3(): # Get macroblock table
    pd.DataFrame(makroblock)
    plot_matrix(makroblock)

def step4(): # Compress macroblock, subtract with 128
    makroblock_subtracted = np.subtract(makroblock, 128)
    pd.DataFrame(makroblock_subtracted)
    plot_matrix(makroblock_subtracted)
    return makroblock_subtracted

def step5(makroblock_subtracted): # Get DCT in macroblock
    makroblock_dct = scipy.fft.dct(makroblock_subtracted)
    pd.options.display.float_format = '{:,.3f}'.format
    pd.DataFrame(makroblock_dct)
    plot_matrix_dct(makroblock_dct)
    return makroblock_dct

def step6(makroblock_dct): # Quantization on macroblock
    quantization_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    makroblock_quantified = np.around(np.divide(
        makroblock_dct, quantization_matrix)).astype(int)
    pd.DataFrame(makroblock_quantified) # 1
    plot_matrix(makroblock_quantified) # 2

    pd.DataFrame(quantization_matrix)
    plot_matrix(makroblock_quantified)

    makroblock_zigzag = zigZag(makroblock_quantified)
    DC = makroblock_zigzag[0]
    AC = makroblock_zigzag[1:]

    print("DC:", DC)
    print("AC:", AC)

def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

def plot_matrix(z):
    # Create the data for the cube
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    # Create the cube
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create the 3D bar chart
    xpos, ypos = np.meshgrid(x, y, indexing='ij')
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    dx = dy = 1  # Width and depth of bars
    dz = z.ravel()

    color1 = '#FFFF00'
    color2 = '#008000'
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, color=get_color_gradient(color1, color2, 64))

    # Set the labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([0, 8])
    ax.set_ylim([0, 8])
    ax.set_zlim([-400, 200])

    # Show the plot
    plt.show()

def plot_matrix_dct(z):
    # Create the data for the cube
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    # Create the cube
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create the 3D bar chart
    xpos, ypos = np.meshgrid(x, y, indexing='ij')
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    dx = dy = 1  # Width and depth of bars
    dz = z.ravel()

    color1 = '#FFFF00'
    color2 = '#008000'
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, color=get_color_gradient(color1, color2, 64))

    # Set the labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([0, 8])
    ax.set_ylim([0, 8])
    ax.set_zlim([-2000, 1000])

    # Show the plot
    plt.show()

def zigZag(a):
    return np.concatenate([np.diagonal(a[::-1,:], k)[::(2*(k % 2)-1)] for k in range(1-a.shape[0], a.shape[0])])

if __name__ == '__main__':
    sample = sys.argv[1]
    img = cv2.imread(sample)
    Y_img, Cr_img, Cb_img = step1()
    makroblock = step2(Y_img)
    step3()
    makroblock_subtracted = step4()
    makroblock_dct = step5(makroblock_subtracted)
    step6(makroblock_dct)
