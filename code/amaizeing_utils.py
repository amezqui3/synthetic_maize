from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as P
import scipy.ndimage as ndimage
import scipy.optimize as optim
from sklearn.metrics import r2_score
import tifffile as tf
import glob
import os
import math

def clean_zeroes(img):
    dim = img.ndim
    orig_size = img.size

    cero = list(range(2*dim))

    for k in range(dim):
        ceros = np.all(img == 0, axis = k)

        for i in range(len(ceros)):
            if(~ceros[i]):
                break
        for j in range(len(ceros)-1, 0, -1):
            if(~ceros[j]):
                break
        cero[k] = i
        cero[k+dim] = j+1

    img = img[cero[1]:cero[3], cero[0]:cero[2]]

    #print(round(100-100*img.size/orig_size),'% reduction from input')

    return img

def plot_hist(img, title='title'):
    hist,bins = np.histogram(img,bins=2**(img.dtype.itemsize*8),range=(0,2**(img.dtype.itemsize*8)))
    hist[0] = 0
    plt.xlabel('intensity')
    plt.ylabel('log(num)')
    plt.title(title)
    plt.grid()
    plt.plot(np.log(hist+1))

def attempt_split(img, sigma=8, threshold=170):
    blur = ndimage.gaussian_filter(img, sigma=8, mode='constant', truncate=3, cval=0)
    img[blur < threshold] = 0
    labels , num = ndimage.label(img, structure=ndimage.generate_binary_structure(img.ndim, 1))
    hist,bins = np.histogram(labels, bins=num, range=(1,num+1))

    return labels, hist, np.sum(hist)

def clean_corner_r(img, tol):
    corner = img[-tol:, -tol:]
    foo = np.where(np.sum(corner, axis=0) > 0)[0]
    while len(foo) > 0 and len(foo) < 0.2*tol:
        bar = np.min(foo + img.shape[1] - tol)
        img[:, bar:] = 0
        img = clean_zeroes(img)
        corner = img[-tol:, -tol:]
        foo = np.where(np.sum(corner, axis=0) > 0)[0]

    corner = img[:tol, -tol:]
    foo = np.where(np.sum(corner, axis=0) > 0)[0]
    while len(foo) > 0 and len(foo) < 0.2*tol:
        bar = np.min(foo + img.shape[1] - tol)
        img[:, bar:] = 0
        img = clean_zeroes(img)
        corner = img[:tol, -tol:]
        foo = np.where(np.sum(corner, axis=0) > 0)[0]
    return img

def clean_corner_l(img, tol):
    corner = img[:tol, :tol]
    foo = np.where(np.sum(corner, axis=0) > 0)[0]
    while len(foo) > 0 and len(foo) < 0.2*tol:
        bar = np.max(foo) + 1
        img[:, :bar] = 0
        img = clean_zeroes(img)
        corner = img[:tol, :tol]
        foo = np.where(np.sum(corner, axis=0) > 0)[0]

    corner = img[-tol:, :tol]
    foo = np.where(np.sum(corner, axis=0) > 0)[0]
    while len(foo) > 0 and len(foo) < 0.2*tol:
        bar = np.max(foo) + 1
        img[:, :bar] = 0
        img = clean_zeroes(img)
        corner = img[-tol:, :tol]
        foo = np.where(np.sum(corner, axis=0) > 0)[0]
    return img

def clean_corner(img, tol=50, sigma=10, thr=175):
    img = clean_corner_l(img, tol)
    img = clean_corner_r(img, tol)

    blur = ndimage.gaussian_filter(img, sigma=10, mode='constant', truncate=3, cval=0)
    img[blur < thr] = 0
    blur = ndimage.gaussian_filter(img, sigma=10, mode='constant', truncate=3, cval=0)
    img[blur < thr] = 0

    labels , num = ndimage.label(img, structure=ndimage.generate_binary_structure(img.ndim, 1))
    hist,bins = np.histogram(labels, bins=num, range=(1,num+1))

    if len(hist) > 1:
        argsort_hist = np.argsort(hist)[::-1]
        regions = ndimage.find_objects(labels)

        i = argsort_hist[0]
        r = regions[i]

        mask = labels[r]==i+1
        box = img[r].copy()
        box[~mask] = 0

        return clean_zeroes(box)
    else:
        return clean_zeroes(img)

def save_comps(dst, foo, img, single, bname='leaf', thr=165, sigma=8, cutoff=1e-2, tol=50, write_file=True):
    labels, hist, sz_hist = attempt_split(foo, sigma, thr)
    leaves = np.sum(hist > cutoff*sz_hist)

    threshold = thr

    if not single:
        while leaves < 2:
            threshold += 3
            labels, hist, sz_hist = attempt_split(foo, sigma, threshold)
            leaves = np.sum(hist > cutoff*sz_hist)

    argsort_hist = np.argsort(hist)[::-1]
    regions = ndimage.find_objects(labels)

    print('Corrected by increasing threshold to', threshold)
    print(leaves, 'leaves')
    print('hist', hist[argsort_hist])

    for j in range(len(regions)):
        i = argsort_hist[j]
        r = regions[i]
        if(hist[i]/sz_hist > cutoff):
            x0,y0,x1,y1 = r[0].start,r[1].start,r[0].stop,r[1].stop
            mask = labels[r]==i+1
            box = img[r].copy()
            box[~mask] = 0
            box = clean_corner(box, tol)
            if write_file:
                tf.imwrite('{}{}_l{}_x{}_y{}.tif'.format(dst,bname,j,x0,y0),
                           box,photometric='minisblack',compress=5)


def separate_leaf(filename, dst, single_leaves, thr=150, sigma=5, cutoff=1e-2, tol=50, w=True):

    src, fname = os.path.split(filename)
    src = src + '/'
    bname = os.path.splitext(fname)[0]
    print(bname)

    single = False
    if bname in single_leaves:
        single = True

    pic = Image.open(filename).convert('L')
    img = np.asarray(pic)
    if img.shape[1] < img.shape[0]:
        img = img.T
    img = img.max() - img
    foo = img.copy()

    foo[foo < thr] = 0

    save_comps(dst, foo, img, single, bname, thr, sigma, cutoff, tol, write_file=w)


# ////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////

# ////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////

def tiff2coords(img, thr = 0, center=True):
    nonzeros = np.sum(img > thr)
    coords = np.empty((nonzeros, img.ndim), dtype=np.float64, order='C')
    idx = 0
    it = np.nditer(img, flags=['multi_index'])
    while not it.finished:
        if it[0] > thr:
            coords[idx, :] = it.multi_index
            idx += 1
        it.iternext()

    if center:
        origin = -1*np.mean(coords, axis=0)
        coords = np.add(coords, origin)

    return coords

def get_margin(img, border):

    surface = ndimage.convolve(img, border, np.int8, 'constant', cval=0)
    surface[ surface < 0 ] = 0
    surface = surface.astype(np.uint8)
    surface[ surface > 0 ] = 1

    labels,num = ndimage.label(surface, structure=ndimage.generate_binary_structure(surface.ndim, 2))
    regions = ndimage.find_objects(labels)

    hist,bins = np.histogram(labels, bins=num, range=(1,num+1))
    sz_hist = np.sum(hist)
    argsort_hist = np.argsort(hist)[::-1]

    i = argsort_hist[0]
    r = regions[i]

    mask = labels[r]==i+1
    box = surface[r].copy()
    box[~mask] = 0

    return box

def orient_base2tip(margin, tol = 100):
    flip = False

    foo,bar = np.nonzero(margin[:,:tol])
    base = np.dstack((foo,bar)).squeeze()

    foo,bar = np.nonzero(margin[:,-tol:])
    tip = np.dstack((foo,bar)).squeeze()

    if tip.shape[0] > base.shape[0]:
        margin = np.flip(margin, 1)
        flip = True

    foo,bar = np.nonzero(margin[:,:tol])
    base = np.dstack((foo,bar)).squeeze()

    foo,bar = np.nonzero(margin[:,-tol:])
    tip = np.dstack((foo,bar)).squeeze()

    return margin, base, tip

def get_midrib(margin):
    midrib = np.empty((margin.shape[1],2))

    for i in range(margin.shape[1]):
        contour = np.nonzero(margin[:,i])[0]
        top , bot = np.max(contour), np.min(contour)
        midrib[i,:] = i, 0.5*(top+bot)

    return midrib

def split_margin(margin, midrib):
    top = []
    bot = []

    for i in range(margin.shape[1]):
        nozero = np.nonzero(margin[:, i])[0]
        top_mask = nozero > midrib[i,1]
        top.append(np.dstack((i*np.ones(len(nozero[top_mask])), nozero[top_mask])).squeeze())
        bot.append(np.dstack((i*np.ones(len(nozero[~top_mask])), nozero[~top_mask])).squeeze())

    return [np.vstack(top) , np.vstack(bot)]

def top_bot_blade(margin, border):
    margin, _ , _ = orient_base2tip(margin)
    midrib = get_midrib(margin)
    t_blade, b_blade = split_margin(margin, midrib)
    coords = np.vstack((t_blade,b_blade))

    return t_blade, b_blade, midrib, np.mean(coords,axis=0)

def refine_margin(margin, tol=0.5, thr=200, keep_tip=0.7, keep_base=0.05):

    threshold = min([thr, int(tol * np.max(np.abs(margin), axis=0)[1])])
    signif = int(keep_tip * margin.shape[0])
    relevant = np.abs(margin[:, 1]) > threshold
    relevant[signif:] = True
    signif = int(keep_base*signif)
    relevant[:signif] = True

    return margin[relevant,:]

def curve_length(x,y):
    curve = np.dstack((x,y)).squeeze()
    lengths = np.sqrt(np.sum(np.diff(curve, axis=0)**2, axis=1)) # Length between corners
    return np.sum(lengths)

def curve_fitting(coords, order=2, skip=0, stepsize=1):

    x_range = np.arange(np.min(coords[skip:,:], axis=0)[0],
                        np.max(coords[skip:,:], axis=0)[0],
                        stepsize)
    poly_fit = P.Polynomial.fit( coords[skip:,0], coords[skip:,1], deg=order, full=False )
    # coeff = poly_fit.convert().coef
    return poly_fit, x_range, poly_fit(x_range)

def poly_margin(margin, blades, tol=0.5, thr=200, keep_tip=0.7, keep_base = 0.05, stepsize=1):
    for i in range(3):
        blades[i] = np.add(blades[i], -1*blades[-1])

    for i in range(2):
        blades[i] = refine_margin(blades[i], tol, thr, keep_tip, keep_base)

    skip = int(keep_base*blades[0].shape[0])

    coef = [None for x in range(3)]
    rang = [None for x in range(len(coef))]
    pred = [None for x in range(len(coef))]

    for i in range(2):
        coef[i], rang[i], pred[i] = curve_fitting(blades[i], 3, skip, stepsize=stepsize)

    base = np.flip(np.vstack((blades[0][:skip,],blades[1][:skip,])), 1)
    coef[2], rang[2], pred[2] = curve_fitting(base, 4, stepsize=stepsize)

    margin = np.vstack((np.dstack((pred[2],rang[2])).squeeze(),
                        np.dstack((rang[0], pred[0])).squeeze(),
                        np.flip(np.dstack((rang[1], pred[1])).squeeze(), axis=0),
                        np.array([pred[2][0],rang[2][0]])))

    return margin, coef, rang, pred

def poly_area(coef, rang):
    area = 0
    for i in range(2):
        integral = coef[i].integ(m=1, k=0)
        area += np.abs(integral(rang[i][-1]) - integral(rang[i][0]))

    i = 2
    integral = coef[i].integ(m=1, k=0)
    #rectangle = np.abs(coef[i].convert().coef[0]*(rang[i][-1]-rang[i][0]))
    rectangle = np.abs(rang[0][0]*(rang[i][-1]-rang[i][0]))

    area += np.abs(integral(rang[i][-1]) - integral(rang[i][0])) - rectangle

    return area

def poly_perimeter(contour, stepsize=1):
    perimeter = 0
    for i in range(len(contour)-1):
        diff = contour[i+1][1] - contour[i][1]
        perimeter += np.sqrt(stepsize*stepsize + diff*diff)

    return perimeter

def summary_shape_size(coef):
    shape = 0
    for i in range(len(coef)):
        shape += len(coef[i]) + 2

    return shape

def summary_model(coef, rang, shape=19):

    summary = np.empty(shape)
    beg = 0
    for i in range(len(coef)):
        end = beg + len(coef[i])
        summary[beg:end] = coef[i].convert().coef
        beg = end + 2
        summary[end:beg] = rang[i][0], rang[i][-1]

    return summary

def model_leaf(tiff_file, border, tol=0.5, thr=200, keep_tip=0.7, keep_base = 0.05):

    img = tf.imread(tiff_file)
    img[img > 0] = 1
    img = ndimage.binary_fill_holes(img)
    if img.shape[0] > img.shape[1]:
        img = img.T

    margin = get_margin(img, border)

    scan_perimeter = np.sum(margin)
    scan_area = np.sum(img)

    blades = top_bot_blade(margin, border)
    contour, coef, rang, pred = poly_margin(margin, list(blades), tol, thr, keep_tip, keep_base)

    model_perimeter = poly_perimeter(contour)
    model_area = poly_area(coef, rang)
    length, width = np.max(contour, axis=0) - np.min(contour, axis=0)

    shape = summary_shape_size(coef)

    summary = np.empty(shape + 6)
    summary[:6] = scan_perimeter, scan_area, model_perimeter, model_area, length, width
    summary[6:] = summary_model(coef, rang, shape)

    return summary

#/////////////////////////////////////////////////////////////////////////////////////////
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#/////////////////////////////////////////////////////////////////////////////////////////
def parabola_fit(phi1, phi3, L):
    b = math.tan(phi1)
    D1 = b
    D2 = 2*(math.tan(phi3)-math.tan(phi1)) + b

    S1 = math.sqrt(1 + D1*D1)
    S2 = math.sqrt(1 + D2*D2)

    K = math.log((D2+S2)/(D1+S1)) + D2*S2 - D1*S1
    a = K/(4*L)

    B = (math.tan(phi3) - math.tan(phi1))/a

    return [0,b,a], B

def catenary_func(alpha, k, t1, t3):
    return k - t3*math.asinh(t1) - t3*math.log(alpha + math.sqrt(1+alpha*alpha)) - math.sqrt(1+alpha*alpha)

def catenary_fit(phi1, phi3, L):

    t1 = math.tan(phi1)
    t3 = math.tan(phi3)
    k = math.sqrt(1 + t1*t1)

    alpha = optim.fsolve(catenary_func, 20, args=(k,t1,t3))

    a = L/(t1 + alpha)
    b = a*math.asinh(t1)
    c = a*k

    B = a*(math.asinh((L-a*t1)/a) + math.asinh(t1))
    return [a,b,c], B

def catenary(x, coefs):
    return -coefs[0]*np.cosh((x-coefs[1])/coefs[0]) + coefs[2]

#/////////////////////////////////////////////////////////////////////////////////////////
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#/////////////////////////////////////////////////////////////////////////////////////////

def param_fit_up(deg, length):
    theta0 = deg[0]/180*math.pi
    theta1 = deg[1]/180*math.pi
    a = length[0]*np.cos(theta1)
    c = length[0]*np.sin(theta1)
    N = a*np.tan(theta0)/c
    A1 = np.power(-1, np.floor(N)+1)*c/np.power(a, np.floor(N))
    A2 = np.power(-1, np.ceil(N)+1)*c/np.power(a, np.ceil(N))

    return A1, A2, N, a, c

def param_fit_down(deg, length, dangle=1.0):
    theta0 = -(dangle*deg[0])/180*math.pi
    theta1 = deg[1]/180*math.pi
    theta2 = deg[2]/180*math.pi
    a = length[0]*np.cos(theta1)
    c = length[0]*np.sin(theta1)
    b = length[1]*np.cos(theta2)
    d = length[1]*np.sin(theta2)

    N = (b-a)*np.tan(theta0)/(d-c)

    A1 = (d-c)/np.power(b-a, np.floor(N))
    A2 = (d-c)/np.power(b-a, np.ceil(N))

    return A1, A2, N, b, d

def polycurve(x, A, N, a, c):
    return A*np.power(x-a, N) + c


def poly_weight_blade_fit(deg, length, resol=50, dangle_correction=True):
    down_curled = False

    A1, A2, N, a, c = param_fit_up(deg, length)
    B1, B2, M, b, d = param_fit_down(deg, length)

    if b <= a or d >= c:
        print('DOWN BIT IS CURLED!')
        B1,B2,M = 0,0,0
        down_curled = True

    if not down_curled and dangle_correction:
        dangle = 1.0
        while M < 1.3:
            dangle += 0.05
            B1, B2, M, b, d = param_fit_down(deg, length, dangle)
        while M > 5.5:
            dangle -= 0.05
            B1, B2, M, b, d = param_fit_down(deg, length, dangle)

    cxrange1 = np.linspace(0, a, 100)
    cxrange2 = np.linspace(a, b, 100)

    floor_up = polycurve(cxrange1, A1, math.floor(N), a, c)
    ceil_up  = polycurve(cxrange1, A2, math.ceil(N), a, c)

    weight_up = np.ceil(N) - N
    weighted_up = weight_up*floor_up + (1.0 - weight_up)*ceil_up

    floor_down = polycurve(cxrange2, B1, math.floor(M), a, c)
    ceil_down  = polycurve(cxrange2, B2, math.ceil(M), a, c)

    weight_down = np.ceil(M) - M
    weighted_down = weight_down*floor_down + (1.0 - weight_down)*ceil_down

    cxrange = np.hstack((cxrange1, cxrange2))
    weighted = np.hstack((weighted_up, weighted_down))

    return np.column_stack((cxrange,weighted)), [A1,N,B1,M, a,b,c,d]

def plot_weighted_results(i, degs, lengths, data, src, writefig=False, verbose=False):
    down_curled = False
    plott = data.iloc[i,0]
    print('Left plot L1 ' + plott)
    deg = np.sort(degs[i])
    length = lengths[i]

    deg = 90 - deg
    theta = deg/180*math.pi

    A1, A2, N, a, c = param_fit_up(deg, length)
    B1, B2, M, b, d = param_fit_down(deg, length)

    if b < (a+0.5) or d >= c:
        print('DOWN BIT IS CURLED!')
        B1,B2,M = -1,-1,-1
        down_curled = True

    if not down_curled:
        dangle = 1.0
        while M < 1.3:
            dangle += 0.1
            B1, B2, M, b, d = param_fit_down(deg, length, dangle)
        while M > 5.5:
            dangle -= 0.1
            B1, B2, M, b, d = param_fit_down(deg, length, dangle)

    if verbose:
        print(deg)
        print(length)
        print('Af = {:.2e}\tAc = {:.2e}\tN = {:.2f}\ta = {:.2f}\tc = {:.2f}'.format(A1,A2,N,a,c))
        print('Bf = {:.2e}\tBc = {:.2e}\tM = {:.2f}\tb = {:.2f}\td = {:.2f}'.format(B1,B2,M,b,d))
        print('*****________*****')

    cxrange1 = np.linspace(0, a, 100)
    cxrange2 = np.linspace(a, b, 100)

    floor_up = polycurve(cxrange1, A1, math.floor(N), a, c)
    ceil_up  = polycurve(cxrange1, A2, math.ceil(N), a, c)

    weight_up = np.ceil(N) - N
    weighted_up = weight_up*floor_up + (1.0 - weight_up)*ceil_up

    if not down_curled:
        floor_down = polycurve(cxrange2, B1, math.floor(M), a, c)
        ceil_down  = polycurve(cxrange2, B2, math.ceil(M), a, c)

        weight_down = np.ceil(M) - M
        weighted_down = weight_down*floor_down + (1.0 - weight_down)*ceil_down

    plt.figure(figsize=(12,10))

    plt.plot(cxrange1, weighted_up, c='green', lw=3, label='weighted')

    if not down_curled:
        plt.plot(cxrange2, weighted_down, c='green', lw=3)

    plt.axline((0,0), slope=math.tan(theta[0]), c='b', ls='--', label='angle {}'.format(deg[0]))
    plt.axline((0,0), slope=math.tan(theta[1]), c='b', ls='--', label='angle {}'.format(deg[1]))
    plt.axline((0,0), slope=math.tan(theta[2]), c='b', ls='--', label='angle {}'.format(deg[2]))
    plt.axvline(x=0, c='k', lw=3)
    plt.axhline(y=0, c='k', lw=3)

    plt.plot([a,a,0],[0,c,c], c='r', ls='dotted')#, marker='v', ms='14')
    plt.plot([b,b,0],[0,d,d], c='r', ls='dotted')#, marker='v', ms='14')

    plt.legend(fontsize=12)
    plt.title('Left plot L1 ' + plott, fontsize=20)
    plt.axis('equal');
    if writefig:
        plt.savefig(src + 'polynomial_model_LPL1_'+plott+'.png', dpi=200, format='png', bbox_inches='tight',
                    facecolor='white', transparent=False)
        plt.close();

    return a,b,c,d,A1,A2,N,B1,B2,M

def show_results(i, degs, lengths, data, src, writefig=False):
    deg = np.sort(degs[i])
    length = lengths[i]

    deg = 90 - deg
    theta = deg/180*math.pi
    print(deg)
    print(length)

    A1, A2, N, a, c = param_fit_up(deg, length)
    print('Af = {:.2e}\tAc = {:.2e}\tN = {:.2f}\ta = {:.2f}\tc = {:.2f}'.format(A1,A2,N,a,c))

    B1, B2, M, b, d = param_fit_down(deg, length)
    print('Bf = {:.2e}\tBc = {:.2e}\tM = {:.2f}\tb = {:.2f}\td = {:.2f}'.format(B1,B2,M,b,d))

    cxrange1 = np.linspace(0, a, 100)
    cxrange2 = np.linspace(a, b, 100)
    plt.figure(figsize=(12,10))


    plt.plot(cxrange1, polycurve(cxrange1, A1, math.floor(N), a, c), c='lime', lw=3, label='floor')
    plt.plot(cxrange2, polycurve(cxrange2, B1, math.floor(M), a, c), c='lime', lw=3)

    plt.plot(cxrange1, polycurve(cxrange1, A2, math.ceil(N), a, c), c='darkgreen', lw=3, label='ceil')
    plt.plot(cxrange2, polycurve(cxrange2, B2, math.ceil(M), a, c), c='darkgreen', lw=3)

    plt.axline((0,0), slope=math.tan(theta[0]), c='b', ls='--', label='angle {}'.format(deg[0]))
    plt.axline((0,0), slope=math.tan(theta[1]), c='b', ls='--', label='angle {}'.format(deg[1]))
    plt.axline((0,0), slope=math.tan(theta[2]), c='b', ls='--', label='angle {}'.format(deg[2]))
    plt.axvline(x=0, c='k', lw=3)
    plt.axhline(y=0, c='k', lw=3)

    plt.axvline(x=a, c='r', ls='dotted')
    plt.axvline(x=b, c='r', ls='dotted')
    plt.axhline(y=c, c='r', ls='dotted')
    plt.axhline(y=d, c='r', ls='dotted')

    plt.legend(fontsize=12)
    plt.title('Left plot L1' + data.iloc[i,1], fontsize=20)
    plt.axis('equal');
    if writefig:
        plt.savefig(src + 'polynomial_model_L_'+data.iloc[i,1]+'.png', dpi=200, format='png', bbox_inches='tight',
                    facecolor='white', transparent=False)
        plt.close();

#/////////////////////////////////////////////////////////////////////////////////////////
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#/////////////////////////////////////////////////////////////////////////////////////////

def blade_fit_up(deg, length, thr=2.25):
    theta0 = deg[0]/180*math.pi
    theta1 = deg[1]/180*math.pi
    a = length[0]*np.cos(theta1)
    c = length[0]*np.sin(theta1)
    N = np.min([thr, a*np.tan(theta0)/c])
    A = c/np.power(a, N)

    return A, N, a, c

def blade_fit_dw(deg, length, dangle=.5):

    theta1 = deg[1]/180*math.pi
    theta2 = deg[2]/180*math.pi
    a = length[0]*np.cos(theta1)
    c = length[0]*np.sin(theta1)
    b = length[1]*np.cos(theta2)
    d = length[1]*np.sin(theta2)

    if ((b-a) <= 3) or ((c-d) <= 1):
        return 0,0,b,d
    if dangle == 0:
        N = 1
    else:
        theta0 = -(dangle*deg[0])/180*math.pi
        N = (b-a)*np.tan(theta0)/(d-c)

    A = (d-c)/np.power(b-a, N)

    return A, N, b, d

def polycurve_up(x, A, N, a, c):
    poly = A*np.power(np.abs(x-a), N) + c
    poly = -poly + poly[0]
    return poly

def polycurve_dw(x, A, N, a, c):
    poly = A*np.power(np.abs(x-a), N) + c
    if N < 1:
        poly = -np.flip(poly) + poly[0] + poly[-1]
    return poly

def poly_blade_fit(deg, length, resol=50, dangle_correction=True, itermax=100, thr=2.25):
    down_curled = False
    dangle = 0.5
    A,N,a,c = blade_fit_up(deg, length, thr)
    if N < 1.1:
        dangle_correction = False
        dangle = 0.0

    B,M,b,d = blade_fit_dw(deg, length, dangle=dangle)

    if B==0 or M==0:
        down_curled = True

    if not down_curled and dangle_correction:
        dangle = .5
        it = 0
        while ((M > N +.25) or (M > 2.5)) and (it < itermax):
            dangle -= 0.01
            B, M, b, d = blade_fit_dw(deg, length, dangle)
            it += 1
        dangle = .5

        it = 0
        while ((M < N -.25) or (M < .25)) and (it < itermax):
            dangle += 0.01
            B, M, b, d = blade_fit_dw(deg, length, dangle)
            it += 1

    cxrange1 = np.linspace(0, a, resol)
    blade_up = polycurve_up(cxrange1, A, N, a, c)

    if not down_curled:
        cxrange2 = np.linspace(a, b, resol)
        blade_dw = polycurve_dw(cxrange2, B, M, a, c)
        cxrange = np.hstack((cxrange1, cxrange2))
        blade = np.hstack((blade_up, blade_dw))
    else:
        cxrange = cxrange1
        blade = blade_up

    return np.column_stack((cxrange,blade)), [A,N,B,M, a,b,c,d], down_curled

def blade_lengths(blade, resol=50, down_curled=False):

    length_blade = np.sum(np.sqrt(np.sum(np.diff(blade, axis=0)**2, axis=1)))
    norm_blade = blade/length_blade

    upblade = blade[:resol]
    length_upblade = np.sum(np.sqrt(np.sum(np.diff(upblade, axis=0)**2, axis=1)))
    norm_upblade = upblade/length_upblade

    if not down_curled:
        dwblade = blade[resol:]
        length_dwblade = np.sum(np.sqrt(np.sum(np.diff(dwblade, axis=0)**2, axis=1)))
        norm_dwblade = dwblade/length_dwblade
    else:
        dwnblade = []
        length_dwblade = 0
        norm_dwblade = []

    return length_blade, length_upblade, length_dwblade


def plot_poly_blade(blades, deg, params, title='title', labels=['label'], writefig=False, dst='./', verbose=False, dpi=100):

    csscolors = ['darkgreen','lawngreen', 'chocolate', 'darkorange','steelblue', 'indigo']
    if len(blades) != len(params):
        print('len(blades) != len(params)')
        return -1

    theta = deg/180*math.pi
    plt.figure(figsize=(12,10))

    for i in range(len(blades)):
        A,N,B,M, a,b,c,d = params[i]
        plt.plot(blades[i][:,0], blades[i][:,1], c=csscolors[i], lw=5, label=labels[i], zorder=i+10)
        if verbose:
            print('\tModel:\t{}'.format(labels[i]))
            print('Up::::\tA = {:.2e}\tN = {:.2f}'.format(A,N))
            print('Down::\tB = {:.2e}\tM = {:.2f}'.format(B,M))

    if verbose:
        print('--------\na = {:.2f}\tc = {:.2f}'.format(a,c))
        print('b = {:.2f}\td = {:.2f}\n-------'.format(b,d))

    plt.axline((0,0), slope=math.tan(theta[0]), c='b', ls='--', label='angle ${}^\circ$'.format(int(90-deg[0])), zorder=1)
    plt.axline((0,0), slope=math.tan(theta[1]), c='b', ls=(0, (3, 5, 1, 5, 1, 5)), label='angle ${}^\circ$'.format(int(90-deg[1])), zorder=2)
    plt.axline((0,0), slope=math.tan(theta[2]), c='b', ls='-.', label='angle ${}^\circ$'.format(int(90-deg[2])), zorder=3)
    plt.axvline(x=0, c='k', lw=3, zorder=4)
    plt.axhline(y=0, c='k', lw=3, zorder=5)

    plt.plot([a,a,0],[0,c,c], c='r', ls='-.', zorder=6)
    plt.plot([b,b,0],[0,d,d], c='r', ls='-.', zorder=7)

    plt.plot([a,b],[c,d], 'or', ms=12, zorder=8)

    plt.legend(fontsize=15)
    plt.title(title, fontsize=20)
    plt.tick_params(labelsize=15)
    plt.axis('equal');

    if writefig:
        filename = '_'.join(title.split(' ')).lower()
        plt.savefig(dst + 'poly_model_'+filename+'.png', dpi=dpi, format='png', bbox_inches='tight',
                    facecolor='white', transparent=False, pil_kwargs={'optimize':True})
        plt.close();


#/////////////////////////////////////////////////////////////////////////////////////////
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#/////////////////////////////////////////////////////////////////////////////////////////

def fit_allometry(data, i, j, log=True, plot_fig=True, init=0, dpi=150, title='Allometry', w=False, dst='./'):
    traits = data.columns[init:]
    #print(traits[i], 'vs', traits[j])

    if log:
        valx = np.abs(data[traits[i]].values)
        valy = np.abs(data[traits[j]].values)
        mask = (valx > 0) & (valy > 0)
        valx = valx[mask]
        valy = valy[mask]

        X = np.log(valx)
        Y = np.log(valy)
    else:
        X = data[traits[i]].values
        Y = data[traits[j]].values

    coef = np.polyfit(X,Y,deg=1)
    lin_fit = np.poly1d(coef)
    R2 = r2_score(Y, lin_fit(X))

    if plot_fig:
        x = np.linspace(np.nanmin(X), np.nanmax(X), 100)

        fig, ax = plt.subplots(figsize=(14,6))

        ax.plot(X, Y, 'ob', alpha=0.1, markersize=7)
        ax.plot(x, lin_fit(x), color='red', lw=6, label='R2 = {:.2f}'.format(R2), alpha=0.8)

        ax.set_title( title + ': ' + traits[j]+' vs '+traits[i], fontsize=26)

        if log:
            ax.set_xlabel('log('+traits[i]+')', fontsize=23)
            ax.set_ylabel('log('+traits[j]+')', fontsize=23)
        else:
            ax.set_xlabel(traits[i], fontsize=23)
            ax.set_ylabel(traits[j], fontsize=23)

        ax.tick_params(labelsize=20)
        ax.legend(fontsize=23, loc='lower right')
        plt.tight_layout()

        if w:
            if log:
                filename = dst + 'log_allometry_' + traits[j]+'_vs_'+traits[i]+'.jpg'
            else:
                filename = dst + 'allometry_' + traits[j]+'_vs_'+traits[i]+'.jpg'
            plt.savefig(filename, dpi=dpi, format='jpg', bbox_inches='tight', pil_kwargs={'optimize':True})
            plt.close()

    return traits[i], traits[j], R2

def residuals(dst,data, i, j, y_pred, init=0, cutoff=20, w=False):
    traits = data.columns[init:]
    X = np.log(data[traits[i]])
    Y = np.log(data[traits[j]])

    residual = np.abs(y_pred - Y)
    foo = np.argsort(residual).values
    plt.axhline(residual[foo[-cutoff]], c='blue')
    plt.scatter(X, residual, alpha=0.1, color='red')
    plt.title('Residual: ' + traits[j]+' vs '+traits[i] )# + ' : ' + str(res))
    plt.xlabel('log('+traits[i]+')')
    plt.ylabel('| Y0 - Y_pred |')
    plt.grid()

    if w:
        filename = dst+'residual_' + traits[j]+'_vs_'+traits[i]+'.jpg'
        plt.savefig(filename, dpi=150, format='jpg', pil_kwargs={'optimize':True})

    plt.close()
    return residual, traits[i], traits[j]

def outliers(dst, data, traits_i, traits_j, residual, cutoff=20):
    foo = np.argsort(residual).values
    outlier = data.iloc[residual[foo[-cutoff:-80]].index]
    outlier.to_csv(dst+'outliers_-_' + traits_j + '_vs_' + traits_i +'.csv',
               columns=['Scan','Label','Color','Tag'], header=False, index=False)
    #examine = pd.DataFrame(outlier.Scan.map(str) +'/l'+ outlier.Label.map(str) +'_'+ outlier.Color.map(str) +'_seeds/' + outlier.Tag.map(str) + '_p7_d4_t120_o7_e1_g3.tif')
    #examine.to_csv(dst+'outliers_-_' + traits_j + '_vs_' + traits_i+'.csv', header=False, index=False)

    return outlier

#/////////////////////////////////////////////////////////////////////////////////////////
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#/////////////////////////////////////////////////////////////////////////////////////////

def shower_thought():
    fs,lw = 28,5

    fig, ax = plt.subplots(2,5, figsize=(20,8))
    x = np.linspace(-5,5,50)

    i = (0,0)
    ax[i].plot(x, np.power(x,2), '-b', lw=lw)
    ax[i].set_title('$x^{N_1}$', fontsize=fs)

    i = (0,1)
    ax[i].plot(x, -0.3*np.power(x,2), '-b', lw=lw)
    ax[i].set_title('$A_1x^{N_1}$', fontsize=fs)

    i = (0,2)
    ax[i].plot(x, -0.3*np.power(x,2) + 5, '-b', lw=lw)
    ax[i].set_title('$A_1x^{N_1} + C$', fontsize=fs)

    i = (0,3)
    ax[i].plot(x+4.1, -0.3*np.power(x,2) + 5, '-b', lw=lw)
    ax[i].set_title('$A_1(x-B)^{N_1} + C$', fontsize=fs)

    xx = np.linspace(-4.1,0,25)
    i = (0,4)
    ax[i].plot(x+4.1, -0.3*np.power(x,2) + 5, '-w', lw=lw)
    ax[i].plot(xx+4.1, -0.3*np.power(xx,2) + 5, '-b', lw=lw)
    ax[i].set_title('base to apex', fontsize=fs)

    i = (1,0)
    ax[i].plot(x, np.power(x,3), '-b', lw=lw)
    ax[i].set_title('$x^{N_2}$', fontsize=fs)

    i = (1,1)
    ax[i].plot(x, -0.05*np.power(x,3), '-b', lw=lw)
    ax[i].set_title('$A_2x^{N_2}$', fontsize=fs)

    i = (1,2)
    ax[i].plot(x, -0.05*np.power(x,3) + 5, '-b', lw=lw)
    ax[i].set_title('$A_2x^{N_2} + {C}$', fontsize=fs)

    i = (1,3)
    ax[i].plot(x+3, -0.05*np.power(x,3) + 5, '-b', lw=lw)
    ax[i].set_title('$A_2(x-B)^{N_2} + C$', fontsize=fs)

    xx = np.linspace(0,4,25)
    i = (1,4)
    ax[i].plot(x+3, -0.05*np.power(x,3) + 5, '-w', lw=lw)
    ax[i].plot(xx+3, -0.05*np.power(xx,3) + 5, '-b', lw=lw)
    ax[i].set_title('apex to tip', fontsize=fs)

    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i,j].tick_params(labelsize=int(fs*.65))
            ax[i,j].axvline(x=0, c='r')
            ax[i,j].axhline(y=0, c='r')
            ax[i,j].set_aspect('equal')

    ax[0,0].set_aspect(0.3)
    ax[1,0].set_aspect(0.1)

    plt.tight_layout()
