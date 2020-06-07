import cv2
import numpy as numpy, math

def jointBilateral():
    noFlash = cv2.imread('test3a.jpg');
    flash = cv2.imread('test3b.jpg');
    nFChannels = cv2.split(noFlash);
    fChannels = cv2.split(flash);
    
    s = 8;
    maskSize = 11;
    
    mask = getMask(maskSize, s);
    result = [];
    
    for i in range(0, 3):
        newChannel = applyBilateral(mask, nFChannels[i], fChannels[i], s);
        newArray =  numpy.array(newChannel, dtype = numpy.uint8 );
        result.append(newArray);
        
    newImg = cv2.merge(result)

    cv2.imshow('result',newImg)
        

def applyBilateral(mask, noFlash, flash, s):
    xlength = noFlash.shape[0];
    print(xlength)
    ylength = noFlash.shape[1];
    print(ylength)
    result = numpy.zeros([xlength, ylength], dtype = int);
    for i in range(0, xlength):
        for j in range(0, ylength):
            centre =[i,j];
            total = applyMask(mask, noFlash, flash, centre, xlength, ylength, s)
            result[i][j] = int(total);
    return result

def applyMask(mask, noFlash, flash, centre, xlength, ylength, s):
    total = 0;
    coeff = 0;
    denom = 0;
    high = int(len(mask)/2)
    low = int(len(mask)/2) - 1
    for i in range(-low, high):
        for j in range(-low, high):
            tryi = centre[0] + i;
            tryj = centre[1] + j;
            if (0 <= tryi <= (xlength -1)) and (0 <= tryj <= (ylength -1)):
                intensityDiff = int(abs(int(flash[tryi][tryj]) - int(flash[centre[0]][centre[1]])))
                coeff = mask[i+high][j+high]*getGauss(0.01,intensityDiff )
                total += coeff*noFlash[tryi][tryj];
                denom += coeff
            else:
                intensityDiff = int(abs(int(flash[centre[0]][centre[1]]) - int(flash[centre[0]][centre[1]])))
                coeff = mask[0+high][0+high]*getGauss(0.01,intensityDiff)
                total += coeff*noFlash[centre[0]][centre[1]];
                denom += coeff
    return (total/denom);
                
            

def getMask(size, s):
    mask = numpy.zeros([size + 1, size +1], dtype = float)
    centre = int(size / 2) + 1
    for i in range(1, size +1):
        for j in range(1, size +1):
            mask[i][j] = getGauss(s, getDist(i,j, centre));
    return (mask);
    
def getDist(i,j, centre):
    iDist = abs(centre-i);
    jDist = abs(centre-j);
    dist = math.sqrt(jDist**2 + iDist**2);
    return dist;

def getGauss(s, x):
    gauss = 1/(math.sqrt(2*math.pi)*s)*math.e**(-(x**2/(2*s**2)))
    return(gauss);
