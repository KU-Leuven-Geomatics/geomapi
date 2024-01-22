# --*-- coding:utf-8 --*--
import numpy as np
import math
import cv2
from scipy import signal

def compute_balance_classes(nodes, number_of_classes:int)->np.array:
    """Compute the balance based on the mask of an image and the number of classes.
    
    Args:
        nodes (_type_): nodes that contain a mask (np.array)
        number_of_classes (int): number of classes.

    Returns:
        balance: percentual number of class presence in among the nodes.
    """
    classbalances= np.empty((1,int(number_of_classes)))
    for n in nodes:
        classes,counts=np.unique(n.mask,return_counts=True)
        classbalance=np.zeros(int(number_of_classes))
        for cl,c in zip(classes,counts):
                classbalance[int(cl)]=c
        classbalances= np.vstack((classbalances,classbalance))
    sums=np.sum(classbalances,axis=0)
    total=np.sum(sums)
    balance=np.round(sums/total, decimals=2)
    return balance

def depth_map_to_hha(C:np.array, D:np.array, RD:np.array) -> np.array:
    """Compute an image depth map to a more comprehensible HHA image according to https://github.com/charlesCXK/Depth2HHA-python
    This HHA normalizes depth and embeds normal information, making the image significantly more informative.
    
    **NOTE**: this take about 9min to compute for a 24Mp image.

    Args:
        C (np.array): Camera matrix
        D (np.array): Depth image, the unit of each element in it is "meter"
        RD (np.array): Raw depth image, the unit of each element in it is "meter"

    Returns:
        np.array(): depth and normal colored image
    """
    missingMask = (RD == 0)
    pc, N, yDir, h, pcRot, NRot = processDepthImage(D * 100, missingMask, C)

    tmp = np.multiply(N, yDir)
    acosValue = np.minimum(1,np.maximum(-1,np.sum(tmp, axis=2)))
    angle = np.array([math.degrees(math.acos(x)) for x in acosValue.flatten()])
    angle = np.reshape(angle, h.shape)

    '''
    Must convert nan to 180 as the MATLAB program actually does. 
    Or we will get a HHA image whose border region is different
    with that of MATLAB program's output.
    '''
    angle[np.isnan(angle)] = 180        


    pc[:,:,2] = np.maximum(pc[:,:,2], 100)
    I = np.zeros(pc.shape)

    # opencv-python save the picture in BGR order.
    I[:,:,2] = 31000/pc[:,:,2]
    I[:,:,1] = h
    I[:,:,0] = (angle + 128-90)

    # print(np.isnan(angle))

    '''
    np.uint8 seems to use 'floor', but in matlab, it seems to use 'round'.
    So I convert it to integer myself.
    '''
    I = np.rint(I)

    # np.uint8: 256->1, but in MATLAB, uint8: 256->255
    I[I>255] = 255
    HHA = I.astype(np.uint8)
    return HHA


def processDepthImage(z, missingMask, C):
    """Helper function of HHA

    Args:
        z (np.array): depth image in 'centimetres'
        missingMask (np.array): a mask
        C (np.array): camera matrix
    """
    yDirParam_angleThresh = np.array([45, 15]) # threshold to estimate the direction of the gravity
    yDirParam_iter = np.array([5, 5])
    yDirParam_y0 = np.array([0, 1, 0])

    normalParam_patchSize = np.array([3, 10])

    X, Y, Z = getPointCloudFromZ(z, C, 1)

    # with open('pd.txt', 'w', encoding='utf-8') as f:
    #     for i in range(X.shape[0]):
    #         for j in range(X.shape[1]):
    #             f.write('{} {} {}\n'.format(str(X[i,j]), str(Y[i,j]), str(Z[i,j])))

    # restore x-y-z position
    pc = np.zeros([z.shape[0], z.shape[1], 3])
    pc[:,:,0] = X
    pc[:,:,1] = Y
    pc[:,:,2] = Z

    N1, b1 = computeNormalsSquareSupport(z/100, missingMask, normalParam_patchSize[0],
    1, C, np.ones(z.shape))
    N2, b2 = computeNormalsSquareSupport(z/100, missingMask, normalParam_patchSize[1],
    1, C, np.ones(z.shape))

    N = N1

    # Compute the direction of gravity
    yDir = getYDir(N2, yDirParam_angleThresh, yDirParam_iter, yDirParam_y0)
    y0 = np.array([[0, 1, 0]]).T
    R = getRMatrix(y0, yDir)

    # rotate the pc and N
    NRot = rotatePC(N, R.T)

    pcRot = rotatePC(pc, R.T)
    h = -pcRot[:,:,1]
    yMin = np.percentile(h, 0)
    if (yMin > -90):
        yMin = -130
    h = h - yMin

    return pc, N, yDir, h,  pcRot, NRot

def getPointCloudFromZ(Z, C, s=1):
    """Clip out a 2R+1 x 2R+1 window at each point and estimate 
        the normal from points within this window. In case the window 
        straddles more than a single superpixel, only take points in the 
        same superpixel as the centre pixel. 

    Args:
        Z (_type_): use depth image and camera matrix to get pointcloud Z is in 'centimetres'
        C (_type_): camera matrix
        s (int, optional): is the factor by which Z has been upsampled. Defaults to 1.

    Returns:
        _type_: _description_
    """
    h, w= Z.shape
    xx, yy = np.meshgrid(np.array(range(w))+1, np.array(range(h))+1)
    # color camera parameters
    cc_rgb = C[0:2,2] * s       # the first two lines of colomn-3, x0 and the y0
    fc_rgb = np.diag(C[0:2,0:2]) * s    # number on the diagonal line
    x3 = np.multiply((xx - cc_rgb[0]), Z) / fc_rgb[0]
    y3 = np.multiply((yy - cc_rgb[1]), Z) / fc_rgb[1]
    z3 = Z
    return x3, y3, z3

def computeNormalsSquareSupport(depthImage, missingMask, R, sc, cameraMatrix, superpixels):
    """Helper function for HHA

    Args:
        depthImage (np.array): in meters
        missingMask (np.array): boolean mask of what data was missing
        R (np.array): radius of clipping
        sc (np.array):  to upsample or not
        cameraMatrix (np.array): intrinsic camera matrix
        superpixels (np.array):  superpixel map to define bounadaries that should not be straddled

    Returns:
        _type_: _description_
    """
    depthImage = depthImage*100     # convert to centi metres
    X, Y, Z = getPointCloudFromZ(depthImage, cameraMatrix, sc)
    Xf = X
    Yf = Y
    Zf = Z
    pc = np.zeros([depthImage.shape[0], depthImage.shape[1], 3])
    pc[:,:,0] = Xf
    pc[:,:,1] = Yf
    pc[:,:,2] = Zf
    XYZf = np.copy(pc)

    # find missing value
    ind = np.where(missingMask == 1)
    X[ind] = np.nan
    Y[ind] = np.nan
    Z[ind] = np.nan

    one_Z = np.expand_dims(1 / Z, axis=2)
    X_Z = np.divide(X, Z)
    Y_Z = np.divide(Y, Z)
    one = np.copy(Z)
    one[np.invert(np.isnan(one[:, :]))] = 1
    ZZ = np.multiply(Z, Z)
    X_ZZ = np.expand_dims(np.divide(X, ZZ), axis=2)
    Y_ZZ = np.expand_dims(np.divide(Y, ZZ), axis=2)

    X_Z_2 = np.expand_dims(np.multiply(X_Z, X_Z), axis=2)
    XY_Z = np.expand_dims(np.multiply(X_Z, Y_Z), axis=2)
    Y_Z_2 = np.expand_dims(np.multiply(Y_Z, Y_Z), axis=2)

    AtARaw = np.concatenate((X_Z_2, XY_Z, np.expand_dims(X_Z, axis=2), Y_Z_2,
                             np.expand_dims(Y_Z, axis=2), np.expand_dims(one, axis=2)), axis=2)

    AtbRaw = np.concatenate((X_ZZ, Y_ZZ, one_Z), axis=2)

    # with clipping
    AtA = filterItChopOff(np.concatenate((AtARaw, AtbRaw), axis=2), R, superpixels)
    Atb = AtA[:, :, AtARaw.shape[2]:]
    AtA = AtA[:, :, :AtARaw.shape[2]]

    AtA_1, detAtA = invertIt(AtA)
    N = mutiplyIt(AtA_1, Atb)

    divide_fac = np.sqrt(np.sum(np.multiply(N, N), axis=2))
    # with np.errstate(divide='ignore'):
    b = np.divide(-detAtA, divide_fac)
    for i in range(3):
        N[:, :, i] = np.divide(N[:, :, i], divide_fac)

    # Reorient the normals to point out from the scene.
    # with np.errstate(invalid='ignore'):
    SN = np.sign(N[:, :, 2])
    SN[SN == 0] = 1
    extend_SN = np.expand_dims(SN, axis=2)
    extend_SN = np.concatenate((extend_SN, extend_SN, extend_SN), axis=2)
    N = np.multiply(N, extend_SN)
    b = np.multiply(b, SN)
    sn = np.sign(np.sum(np.multiply(N, XYZf), axis=2))
    sn[np.isnan(sn)] = 1
    sn[sn == 0] = 1
    extend_sn = np.expand_dims(sn, axis=2)
    N = np.multiply(extend_sn, N)
    b = np.multiply(b, sn)
    return N, b

def filterItChopOff(f, r, sp):
    """Helper function for HHA

    Args:
        f (_type_): _description_
        r (_type_): _description_
        sp (_type_): _description_

    Returns:
        _type_: _description_
    """
    f[np.isnan(f)] = 0
    H, W, d = f.shape
    B = np.ones([2 * r + 1, 2 * r + 1])     # 2r+1 * 2r+1 neighbourhood

    minSP = cv2.erode(sp, B, iterations=1)
    maxSP = cv2.dilate(sp, B, iterations=1)

    ind = np.where(np.logical_or(minSP != sp, maxSP != sp))

    spInd = np.reshape(range(np.size(sp)), sp.shape,'F')

    delta = np.zeros(f.shape)
    delta = np.reshape(delta, (H * W, d), 'F')
    f = np.reshape(f, (H * W, d),'F')

    # calculate delta

    I, J = np.unravel_index(ind, [H, W], 'C')
    for i in range(np.size(ind)):
        x = I[i]
        y = J[i]
        clipInd = spInd[max(0, x - r):min(H-1, x + r), max(0, y - r):min(W-1, y + r)]
        diffInd = clipInd[sp[clipInd] != sp[x, y]]
        delta[ind[i], :] = np.sum(f[diffInd, :], 1)
    delta = np.reshape(delta, (H, W, d), 'F')
    f = np.reshape(f, (H, W, d), 'F')
    fFilt = np.zeros([H, W, d])

    for i in range(f.shape[2]):
        #  fFilt(:,:,i) = filter2(B, f(:,:,i));
        tmp = cv2.filter2D(np.rot90(f[:, :, i], 2), -1, np.rot90(np.rot90(B, 2), 2))
        tmp = signal.convolve2d(np.rot90(f[:, :, i], 2), np.rot90(np.rot90(B, 2), 2), mode="same")
        fFilt[:, :, i] = np.rot90(tmp, 2)
    fFilt = fFilt - delta
    return fFilt

def mutiplyIt(AtA_1, Atb):
    """Helper function for HHA

    Args:
        AtA_1 (np.array): 
        Atb (np.array): 

    Returns:
        (np.array):  result
    """
    result = np.zeros([Atb.shape[0], Atb.shape[1], 3])
    result[:, :, 0] = np.multiply(AtA_1[:, :, 0], Atb[:, :, 0]) + np.multiply(AtA_1[:, :, 1],
                                                                              Atb[:, :, 1]) + np.multiply(
        AtA_1[:, :, 2], Atb[:, :, 2])
    result[:, :, 1] = np.multiply(AtA_1[:, :, 1], Atb[:, :, 0]) + np.multiply(AtA_1[:, :, 3],
                                                                              Atb[:, :, 1]) + np.multiply(
        AtA_1[:, :, 4], Atb[:, :, 2])
    result[:, :, 2] = np.multiply(AtA_1[:, :, 2], Atb[:, :, 0]) + np.multiply(AtA_1[:, :, 4],
                                                                              Atb[:, :, 1]) + np.multiply(
        AtA_1[:, :, 5], Atb[:, :, 2])
    return result

def invertIt(AtA):
    """Helper function for HHA

    Args:
        AtA (np.array): 

    Returns:
        (np.array,np.array):  AtA_1, detAta
    """
    AtA_1 = np.zeros([AtA.shape[0], AtA.shape[1], 6])
    AtA_1[:, :, 0] = np.multiply(AtA[:, :, 3], AtA[:, :, 5]) - np.multiply(AtA[:, :, 4], AtA[:, :, 4])
    AtA_1[:, :, 1] = -np.multiply(AtA[:, :, 1], AtA[:, :, 5]) + np.multiply(AtA[:, :, 2], AtA[:, :, 4])
    AtA_1[:, :, 2] = np.multiply(AtA[:, :, 1], AtA[:, :, 4]) - np.multiply(AtA[:, :, 2], AtA[:, :, 3])
    AtA_1[:, :, 3] = np.multiply(AtA[:, :, 0], AtA[:, :, 5]) - np.multiply(AtA[:, :, 2], AtA[:, :, 2])
    AtA_1[:, :, 4] = -np.multiply(AtA[:, :, 0], AtA[:, :, 4]) + np.multiply(AtA[:, :, 1], AtA[:, :, 2])
    AtA_1[:, :, 5] = np.multiply(AtA[:, :, 0], AtA[:, :, 3]) - np.multiply(AtA[:, :, 1], AtA[:, :, 1])

    x1 = np.multiply(AtA[:, :, 0], AtA_1[:, :, 0])
    x2 = np.multiply(AtA[:, :, 1], AtA_1[:, :, 1])
    x3 = np.multiply(AtA[:, :, 2], AtA_1[:, :, 2])

    detAta = x1 + x2 + x3
    return AtA_1, detAta

def getYDir(N, angleThresh, iter, y0):
    """Helper function for HHA.

    Args:
        N (_type_): HxWx3 matrix with normal at each pixel.
        angleThresh (_type_): in degrees the threshold for mapping to parallel to gravity and perpendicular to gravity
        iter (_type_): number of iterations to perform
        y0 (_type_): the initial gravity direction

    Returns:
        _type_: _description_
    """
    y = y0
    for i in range(len(angleThresh)):
        thresh = np.pi * angleThresh[i] / 180   # convert it to radian measure
        y = getYDirHelper(N, y, thresh, iter[i])
    return y

def getYDirHelper(N, y0, thresh, num_iter):
    """Helper function for HHA.

    Args:
        N (_type_): HxWx3 matrix with normal at each pixel.
        angleThresh (_type_): in degrees the threshold for mapping to parallel to gravity and perpendicular to gravity
        iter (_type_): number of iterations to perform
        y0 (_type_): the initial gravity direction

    Returns:
        _type_: _description_
    """
    dim = N.shape[0] * N.shape[1]

    # change the third dimension to the first-order. (480, 680, 3) => (3, 480, 680)
    nn = np.swapaxes(np.swapaxes(N,0,2),1,2)
    nn = np.reshape(nn, (3, dim), 'F')

    # remove these whose number is NAN
    idx = np.where(np.invert(np.isnan(nn[0,:])))[0]
    nn = nn[:,idx]

    # Set it up as a optimization problem
    yDir = y0
    for i in range(num_iter):
        sim0 = np.dot(yDir.T, nn)
        indF = abs(sim0) > np.cos(thresh)       # calculate 'floor' set.    |sin(theta)| < sin(thresh) ==> |cos(theta)| > cos(thresh)
        indW = abs(sim0) < np.sin(thresh)       # calculate 'wall' set.
        if(len(indF.shape) == 2):
            NF = nn[:, indF[0,:]]
            NW = nn[:, indW[0,:]]
        else:
            NF = nn[:, indF]
            NW = nn[:, indW]
        A = np.dot(NW, NW.T) - np.dot(NF, NF.T)
        b = np.zeros([3,1])
        c = NF.shape[1]
        w,v = np.linalg.eig(A)      # w:eigenvalues; v:eigenvectors
        min_ind = np.argmin(w)      # min index
        newYDir = v[:,min_ind]
        yDir = newYDir * np.sign(np.dot(yDir.T, newYDir))
    return yDir

def getRMatrix(yi, yf):
    """getRMatrix: Generate a rotation matrix that
            if yf is a scalar, rotates about axis yi by yf degrees
            if yf is an axis, rotates yi to yf in the direction given by yi x yf

    Args:
        yi (np.array): yi is an axis 3x1 vector
        yf (np.array): yf could be a scalar of axis

    Returns:
        _type_: _description_
    """
    if (np.isscalar(yf)):
        ax = yi / np.linalg.norm(yi)        # norm(A) = max(svd(A))
        phi = yf
    else:
        yi = yi / np.linalg.norm(yi)
        yf = yf / np.linalg.norm(yf)
        ax = np.cross(yi.T, yf.T).T
        ax = ax / np.linalg.norm(ax)
        # find angle of rotation
        phi = np.degrees(np.arccos(np.dot(yi.T, yf)))

    if (abs(phi) > 0.1):
        phi = phi * (np.pi / 180)
        ax=ax.flatten()
        s_hat = np.array([[0, -ax[2], ax[1]],
                          [ax[2], 0, -ax[0]],
                          [-ax[1], ax[0], 0]])
        R = np.eye(3) + np.sin(phi) * s_hat + (1 - np.cos(phi)) * np.dot(s_hat, s_hat)      # dot???
    else:
        R = np.eye(3)
    return R

def rotatePC(pc, R):
    """Calibration of gravity direction 

    Args:
        pc (_type_): _description_
        R (_type_): _description_

    Returns:
        _type_: _description_
    """
    if(np.array_equal(R, np.eye(3))):
        return pc
    else:
        R = R.astype(np.float64)
        dim = pc.shape[0] * pc.shape[1]
        pc = np.swapaxes(np.swapaxes(pc, 0, 2), 1, 2)
        res = np.reshape(pc, (3, dim), 'F')
        res = np.dot(R, res)
        res = np.reshape(res, pc.shape, 'F')
        res = np.swapaxes(np.swapaxes(res, 0, 1), 1, 2)
        return res