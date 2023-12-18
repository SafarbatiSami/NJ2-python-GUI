import numpy as np
from tifffile import imwrite, transpose_axes
import os
from skimage import draw

img = np.zeros((100,100,100))

center = (50,50,50)
center = np.array(center)

ray = 50

#---------------------------------------------------------------------------

def gen_ellipsoid(img, center, ray, return_raw=False, a=1, b=1, c=1):
    x,y,z = np.meshgrid(*[np.arange(d) for d in img.shape])
    x,y,z = x-center[0],y-center[1],z-center[2]
    if return_raw:
        msk = np.copy(img)
        img=np.maximum(ray-np.sqrt(x*x/(a*a)+y*y/(b*b)+z*z/(c*c)),0)
        msk[np.sqrt(x*x/(a*a)+y*y/(b*b)+z*z/(c*c))<ray]=1
        return img, msk
    else:
        img[np.sqrt(x*x/(a*a)+y*y/(b*b)+z*z/(c*c))<ray]=1
        return img

def alea_sphere(img_size, range_ray, range_center, return_raw=False):
    img_size = np.array(img_size)
    range_ray = np.array(range_ray)
    range_center = np.array(range_center)

    ray = np.random.randint(range_ray[0], range_ray[1])
    center = np.array([np.random.randint(rc[0], rc[1]) for rc in range_center])
    center = np.where(np.greater(center + ray,img_size), img_size-ray, center)
    center = np.where(np.less(center-ray, 0), ray, center)

    img = np.zeros(img_size)

    a=0.2
    b=1.0
    c=1.6
    return *gen_ellipsoid(img, center, ray, return_raw=return_raw, a=a, b=b, c=c), 4*np.pi*(ray**3*a*b*c)/3

def sphere_dir(path, path_msk, n=20):
    # if os.path.exists(path): os.remove(path)
    # if os.path.exists(path_msk): os.remove(path_msk)
    os.makedirs(path, exist_ok=True)
    os.makedirs(path_msk, exist_ok=True)

    for i in range(n):
        img, msk, vol = alea_sphere((100,100,100),(15,16),((20,80),(20,80),(20,80)), return_raw=True)

        img = (img * 255 / img.max()).astype(int)
        msk = (msk * 255 / msk.max()).astype(int)

        img = transpose_axes(img, 'ZYX', 'TZCYXS')
        msk = transpose_axes(msk, 'ZYX', 'TZCYXS')

        # res = ((10320,1000),(10320,1000),'MICROMETER')
        # metadata = {'spacing':0.2, 'unit':'MICROMETER'}
        # vol = vol * 0.1032 * 0.1032 * 0.2

        res = ((10000,1000),(10000,1000),'MICROMETER')
        metadata = {'spacing':0.2, 'unit':'MICROMETER'}

        name = str(vol+np.random.rand())+'.tif'

        imwrite(os.path.join(path, name), img.astype(np.uint8),
            resolution=res,
            imagej=True, 
            metadata=metadata,
            compression=('zlib', 1))
        imwrite(os.path.join(path_msk, name), msk.astype(np.uint8),
            resolution=res,
            imagej=True, 
            metadata=metadata,
            compression=('zlib', 1))
        print("Measured volume:", np.sum(msk))
        print("Actual volume:", vol)

sphere_dir("data\\img", "data\\msk")
# msk = gen_sphere(img, center, ray)
# msk, vol = alea_sphere((100,100,100),(10,30),((20,80),(20,80),(20,80)))
# imsave("tmp.tif", msk)
# print(len(np.meshgrid(np.arange(10), np.arange(10), np.arange(10))))

#---------------------------------------------------------------------------

