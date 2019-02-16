'''
    PLEASE CHANGE THE WORKING DIRECTORY TO ~/anbhatta_project01/
'''


import cv2
import numpy as np
import math
import time

#LoG function
def log(sigma,fsize):
    l = np.zeros((fsize,fsize))
    m = int(fsize/2)
    for i in range(-m,m + 1):
        for j in range(-m,m+1):
            x = -((math.pow(i, 2) + math.pow(j,2)) / (2 * math.pow(sigma, 2)))
            l[i+m][j+m] = (-1/(math.pi*math.pow(sigma,2)))*(1-((math.pow(i,2)+math.pow(j,2))/(2*math.pow(sigma,2))))* math.exp(x)
    return l

#convolution function
def convolution(gray, filter):
    m,n = filter.shape
    r,c = gray.shape
    conv = np.zeros((r, c))
    if m == n:
        A = np.pad(gray, (m //2, m //2), 'constant')
        for x in range(0,len(gray)):
            for y in range(0,len(gray[0])):
                conv[x][y] = (A[x:x+m, y:y+m]*filter).sum()
    return conv

#non maximum suppression
def non_max(prev,curr,next,th):
    val = []
    for x in range(1, curr.shape[0] - 1):
        for y in range(1, curr.shape[1] - 1):
            a = max(curr[x - 1:x + 2, y - 1:y + 2].max(axis=1))
            b = max(next[x - 1:x + 2, y - 1:y + 2].max(axis=1))
            c = max(prev[x - 1:x + 2, y - 1:y + 2].max(axis=1))
            if max(a,b,c) == curr[x][y]:
                if abs(curr[x][y]) > th:
                    val.append((y,x))
    return val

#program code
def run(gray,img,th):
    level = 14
    sigma = 1 / math.sqrt(2)
    k = 1.24
    sigma_l = []
    scale = []
    sigma_l.append(sigma)
    filter_list = []

    for i in range(1, level + 1):
        sigma_l.append(k * sigma)
        sigma = k * sigma
    for i in range(0, len(sigma_l)):
        fsize = 6 * sigma_l[i]
        if int(fsize) % 2 == 0:
            fsize = fsize + 1
        filter = log(sigma_l[i], int(fsize))
        filter_list.append(filter)
        conv = convolution(gray, filter)
        conv1 = conv ** 2
        scale.append(conv1)

    p = []
    q = []
    img[:, :, 0] = img[:, :, 2]
    img[:, :, 1] = img[:, :, 2]
    scale = np.array(scale)
    
    for x in range(1, scale[0].shape[0] - 1):
        for y in range(1, scale[0].shape[1] - 1):
            a = max(scale[0][x - 1:x + 2, y - 1:y + 2].max(axis=1))
            b = max(scale[1][x - 1:x + 2, y - 1:y + 2].max(axis=1))
            if max(a, b) == scale[0][x][y]:
                if abs(scale[0][x][y]) > th:
                    p.append((y, x))
    for k in range(len(p)):
        cv2.circle(img, p[k], int(sigma_l[0] * math.sqrt(2)), (0, 0, 255), 1)
    for x in range(1, scale[len(scale) - 1].shape[0] - 1):
        for y in range(1, scale[len(scale) - 1].shape[1] - 1):
            a = max(scale[len(scale) - 1][x - 1:x + 2, y - 1:y + 2].max(axis=1))
            b = max(scale[len(scale) - 2][x - 1:x + 2, y - 1:y + 2].max(axis=1))
            if max(a, b) == scale[len(scale) - 1][x][y]:
                if abs(scale[len(scale) - 1][x][y]) > th:
                    q.append((y, x))
    for k in range(len(q)):
        cv2.circle(img, q[k], int(sigma_l[len(sigma_l) - 1] * math.sqrt(2)), (0, 0, 255), 1)
    for i in range(1, len(scale) - 1):
        l = non_max(scale[i - 1], scale[i], scale[i + 1],th)
        radius = int(sigma_l[i] * math.sqrt(2))
        for k in range(len(l)):
            cv2.circle(img, l[k], radius, (0, 0, 255), 1)

    time_end = time.process_time() - time_start
    print('time: ', time_end)

    return img


images = ['sunflowers.jpg', 'butterfly.jpg','einstein.jpg','fishes.jpg','pic.jpg','husky.jpg','redflower.jpg','bird.jpg']

image = input("enter image name from the folder: eg. sunflowers.jpg, einstein.jpg etc ")
if image not in images:
    print("incorrect response, type correctly ")

time_start = time.process_time()
img = cv2.imread(image)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = gray / 255
t = 0.02 #threshold chosen
final = run(gray,img,t)
cv2.imshow('image', final)
cv2.waitKey(0)
cv2.destroyAllWindows()


