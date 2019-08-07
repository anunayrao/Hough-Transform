import cv2
import numpy as np
from math import pi,e
from numpy import unravel_index
np.set_printoptions(threshold=np.nan)

import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
#se = [[1,1,1],[1,1,1],[1,1,1]]
#se = np.asarray(se)
img = cv2.imread("hough.jpg",0)
height,width = img.shape
pmax = int((height**2 + width**2)**0.5)


#Function to generate Gaussian kernel of any size.
def kernel(size,s):
	j=0
	i=0
	c = 0
	d = 0
	g_kernel = [[ 0 for i in range(size)] for j in range(size)]
	y = j+ int(size/2) + 1
	for i in range(size):
		x = -int(size/2)
		y = y - 1
		for j in range(size):
			gauss = (1/(2*pi*s*s))*e**(-0.5*((float(x*x) + float(y*y))/(s*s)))
			g_kernel[i][j] = gauss
			x = x + 1
	for i in range(size):
		for j in range(size):
			c = c + g_kernel[i][j]
			
	co = (float)(1/c)
	
	for i in range(size):
		for j in range(size):
			g_kernel[i][j] = float(co*g_kernel[i][j])
	
	for i in range(size):
		for j in range(size):
			d = d + g_kernel[i][j]
					
	k_arr = np.asarray(g_kernel)
	return k_arr


def blur(img, s,height,width):
	a = [[ 0 for x in range(width+6)] for w in range(height+6)]
	b = [[ 0 for x in range(width)] for w in range(height)]
	
	for i in range(height):
	    for j in range(width):
	        a[i+3][j+3] = img[i][j]

      	  
	a = np.asarray(a)
	ker_arr = kernel(7,s)
	for i in range(height):
	    for j in range(width):
	        sum1=0
	        for k in range(7):
	            for l in range(7):
	                sum1 += ker_arr[k][l]*a[i+k][j+l]
	        b[i][j] = sum1     
	fin = np.asarray(b).astype(np.uint8)
	return fin
	
#gimg = blur(img,1.414,height,width)

gimg = cv2.imread("hough.jpg",0)
se = [[1,1],[1,1]]
se = np.asarray(se)

#cv2.imshow("image",gimg)
#cv2.waitKey(0)

pgimg = [[ 0 for x in range(width + 2)] for w in range(height + 2)]
b = [[ 0 for x in range(width)] for w in range(height)]
c = [[ 0 for x in range(width)] for w in range(height)]
for i in range(height):
    for j in range(width):
        pgimg[i+1][j+1]= gimg[i][j]

sobel_H = [[-1,0,1],[-2,0,2],[-1,0,1]] 
sobel_V = [[-1,-2,-1],[0,0,0],[1,2,1]]

k,l = 1,1

#Applying Sobel operator
for k in range(height+1):
    for l in range(width+1):
        b[k-1][l-1] = pgimg[k-1][l-1]*sobel_H[0][0] + pgimg[k-1][l]*sobel_H[0][1] + pgimg[k-1][l+1]*sobel_H[0][2] + pgimg[k][l-1]*sobel_H[1][0] + pgimg[k][l]*sobel_H[1][1] + pgimg[k][l+1]*sobel_H[1][2] + pgimg[k+1][l-1]*sobel_H[2][0] + pgimg[k+1][l]*sobel_H[2][1] + pgimg[k+1][l+1]*sobel_H[2][2]
        c[k-1][l-1] = pgimg[k-1][l-1]*sobel_V[0][0] + pgimg[k-1][l]*sobel_V[0][1] + pgimg[k-1][l+1]*sobel_V[0][2] + pgimg[k][l-1]*sobel_V[1][0] + pgimg[k][l]*sobel_V[1][1] + pgimg[k][l+1]*sobel_V[1][2] + pgimg[k+1][l-1]*sobel_V[2][0] + pgimg[k+1][l]*sobel_V[2][1] + pgimg[k+1][l+1]*sobel_V[2][2]

#Find maximum 
max =0
for i in range(height):
	for j in range(width):
		if(max<b[i][j]):
			max = b[i][j]

for i in range(height):
	for j in range(width):
		if(b[i][j]<0):
			b[i][j] = b[i][j]*(-1)
		if(c[i][j]<0):
			c[i][j] = c[i][j]*(-1)

b = np.asarray(b)		
c = np.asarray(c)	

#Eliminating values
pos_edge_x=(b) /(max)
pos_edge_y=(c)/(max)

#Combining both X and Y
edge_magnitude = (pos_edge_x ** 2 + pos_edge_y ** 2)**(0.5)
edge_magnitude2 = (pos_edge_x ** 2 + pos_edge_y ** 2)**(0.5)

#edge = edge_magnitude*255
#cv2.imwrite('abc.jpg',edge)

m =0
for i in range(height):
	for j in range(width):
		if(edge_magnitude[i][j]>0.25):
			edge_magnitude2[i][j] = 255
		else:
			edge_magnitude2[i][j] = 0
#edge_magnitude /= m
edge_magnitude2[0:10,:] = 0
edge_magnitude2[:,0:10] = 0
edge_magnitude2[:, width-10:width] = 0   
edge_magnitude2[height-10:height,:] = 0

#edge_magnitude2 /=255
#cv2.imshow('image_Combined',edge_magnitude2)


#print(edge_magnitude)

theta = np.linspace(0,90,91, dtype=np.int)
rho = np.linspace(0,pmax,164,dtype=np.int)
H = np.zeros((len(theta),len(rho)), dtype=np.int)

for x in range(width):
    for y in range(height):
        if(edge_magnitude2[y][x]>0):
            for i in range(35,39):
                t = theta[i]*pi/180
                d= x*(np.cos(t)) + (height-y)*np.sin(t)
                r = abs(rho-d)
                dr = min(r)
                ir = r.argmin()
                if(dr<=1):
                    H[i][ir] = H[i][ir]+1
#print(H[36])
#plt.plot(rho, H[36])
#plt.show()
rhoind = H[36]
rhof = []
thresh = 7
for i in range(len(rhoind)-2):
    if((rhoind[i+1]-rhoind[i])>thresh and rhoind[i+1]-rhoind[i+2]>thresh):
        rhof.append(i+1)
        thresh = 30 
      
angle = 36
a = np.cos((90-angle)*pi/180)
b = np.sin((90-angle)*pi/180)
image = cv2.imread("hough.jpg",1)
for i in rhof:
    x0 = a*rho[i]
    y0 = b*rho[i]
    x1 = int(x0 +1000*(b))
    y1 = int(y0 +1000*(-a))
    x2 = int(x0 -1000*(b))
    y2 = int(y0 -1000*(-a))
    cv2.line(image,(y1,height-x1),(y2,height-x2),(255,0,0),2)
    #cv2.line(image,(y1,height-x1),(y2,height-x2),(0,0,0),2)
#cv2.imshow("Blue Line detection",image)
cv2.imwrite('blue_lines.jpg',image)
#cv2.waitKey(0)



###########################################################################

theta = np.linspace(0,90,91, dtype=np.int)
rho = np.linspace(0,pmax,164,dtype=np.int)
H = np.zeros((len(theta),len(rho)), dtype=np.int)

for x in range(width):
    for y in range(height):
        if(edge_magnitude2[y][x]>0):
            for i in range(0,5):
                t = theta[i]*pi/180
                d= x*(np.cos(t)) + (height-y)*np.sin(t)
                r = abs(rho-d)
                dr = min(r)
                ir = r.argmin()
                if(dr<=1):
                    H[i][ir] = H[i][ir]+1
#print(H[2])
#plt.plot(rho, H[2])
#plt.show()
rhoind = H[2]
rhof = []
thresh = 30
for i in range(len(rhoind)-2):
    if((rhoind[i+1]-rhoind[i])>thresh and rhoind[i+1]-rhoind[i+2]>thresh and rhoind[i]!=0):
        rhof.append(i+1)
        thresh = 30 
      
angle = 2
a = np.cos((90-angle)*pi/180)
b = np.sin((90-angle)*pi/180)
image = cv2.imread("hough.jpg",1)
for i in rhof:
    x0 = a*rho[i]
    y0 = b*rho[i]
    x1 = int(x0 +1000*(b))
    y1 = int(y0 +1000*(-a))
    x2 = int(x0 -1000*(b))
    y2 = int(y0 -1000*(-a))
    cv2.line(image,(y1,height-x1),(y2,height-x2),(0,0,255),2)
    #cv2.line(image,(y1,height-x1),(y2,height-x2),(0,0,0),2)
#for j in rhof:
	#print(rho[j])
#cv2.imshow("Red Line detection",image)
cv2.imwrite('red_lines.jpg',image)
#cv2.waitKey(0)


#######################################################################

def detectCircles(img,threshold,region):
    rmax= 30
    rmin= 20

    R = rmax - rmin
    A = np.zeros((rmax,height+2*rmax,width+2*rmax))
    B = np.zeros((rmax,height+2*rmax,width+2*rmax))
    theta = np.arange(0,360)*np.pi/180
    edges = np.argwhere(img[:,:])                                             
    for val in range(R):
        r = rmin+val
        bprint = np.zeros((2*(r+1),2*(r+1)))
        (m,n) = (r+1,r+1)                                                     
        for angle in theta:
            x = int(np.round(r*np.cos(angle)))
            y = int(np.round(r*np.sin(angle)))
            bprint[m+x,n+y] = 1
        constant = np.argwhere(bprint).shape[0]
        for x,y in edges:                                                       
            X = [x-m+rmax,x+m+rmax]                                          
            Y= [y-n+rmax,y+n+rmax]                                           
            A[r,X[0]:X[1],Y[0]:Y[1]] += bprint
        A[r][A[r]<threshold*constant/r] = 0

    for r,x,y in np.argwhere(A):
        temp = A[r-region:r+region,x-region:x+region,y-region:y+region]
        try:
            p,a,b = np.unravel_index(np.argmax(temp),temp.shape)
        except:
            continue
        B[r+(p-region),x+(a-region),y+(b-region)] = 1

    return B[:,rmax:-rmax,rmax:-rmax]

def displayCircles(A):
    img = cv2.imread('hough.jpg')
    fig = plt.figure()
    plt.imshow(img)
    circleCoordinates = np.argwhere(A)                             
    circle = []
    for r,x,y in circleCoordinates:
        circle.append(plt.Circle((y,x),r,color=(1,0,0),fill=False))
        fig.add_subplot(111).add_artist(circle[-1])
    plt.savefig('coins.jpg')
    plt.show()
        
res = detectCircles(edge_magnitude2,8.2,40)
displayCircles(res)