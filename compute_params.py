import argparse
import os
import subprocess
import sys
import pathlib

def install(packages):
    """Install a list of packages.
    """
    for p in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", p])

try:
    import tifffile as tiff
    from skimage import measure
    from skimage import io
    from skimage.transform import resize
    from scipy.spatial import Delaunay
    import numpy as np
    import pandas as pd

except ImportError as e:
    print("[Warning] Some packages are missing. Installing...")
    install(['tifffile', 'scikit-image', 'scipy', 'numpy', 'pandas'])
    import tifffile as tiff
    from skimage import measure
    from skimage import io
    from skimage.transform import resize
    from scipy.spatial import Delaunay
    import numpy as np
    import pandas as pd

#---------------------------------------------------------------------------
# FOR DEBUGGING ONLY
# Resizing

def resize_segmentation(segmentation, new_shape, order=3):
    '''
    Copied from batch_generator library. Copyleft Fabian Insensee.
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped

def resize_3d(img, output_shape, order=3, is_msk=False, monitor_anisotropy=True, anisotropy_threshold=3):
    """
    Resize a 3D image given an output shape.
    
    Parameters
    ----------
    img : numpy.ndarray
        3D image to resample.
    output_shape : tuple, list or numpy.ndarray
        The output shape. Must have an exact length of 3.
    order : int
        The order of the spline interpolation. For images use 3, for mask/label use 0.

    Returns
    -------
    new_img : numpy.ndarray
        Resized image.
    """
    assert len(img.shape)==4, '[Error] Please provided a 3D image with "CWHD" format'
    assert len(output_shape)==3 or len(output_shape)==4, '[Error] Output shape must be "CWHD" or "WHD"'
    
    # convert shape to array
    input_shape = np.array(img.shape)
    output_shape = np.array(output_shape)
    if len(output_shape)==3:
        output_shape = np.append(input_shape[0],output_shape)
    if np.all(input_shape==output_shape): # return image if no reshaping is needed
        return img 
        
    # resize function definition
    resize_fct = resize_segmentation if is_msk else resize
    resize_kwargs = {} if is_msk else {'mode': 'edge', 'anti_aliasing': False}
        
    # separate axis --> [Guillaume] I am not sure about the interest of that... 
    # we only consider the following case: [147,512,513] where the anisotropic axis is undersampled
    # and not: [147,151,512] where the anisotropic axis is oversampled
    anistropy_axes = np.array(input_shape[1:]) / input_shape[1:].min()
    do_anisotropy = monitor_anisotropy and len(anistropy_axes[anistropy_axes>anisotropy_threshold])==2
    if not do_anisotropy:
        anistropy_axes = np.array(output_shape[1:]) / output_shape[1:].min()
        do_anisotropy = monitor_anisotropy and len(anistropy_axes[anistropy_axes>anisotropy_threshold])==2
        
    do_additional_resize = False
    if do_anisotropy: 
        axis = np.argmin(anistropy_axes)
        print("[resize] Anisotropy monitor triggered! Anisotropic axis:", axis)
        
        # as the output_shape and the input_shape might have different dimension
        # along the selected axis, we must use a temporary image.
        tmp_shape = output_shape.copy()
        tmp_shape[axis+1] = input_shape[axis+1]
        
        tmp_img = np.empty(tmp_shape)
        
        length = tmp_shape[axis+1]
        tmp_shape = np.delete(tmp_shape,axis+1)
        
        for c in range(input_shape[0]):
            coord  = [c]+[slice(None)]*len(input_shape[1:])

            for i in range(length):
                coord[axis+1] = i
                tmp_img[tuple(coord)] = resize_fct(img[tuple(coord)], tmp_shape[1:], order=order, **resize_kwargs)
            
        # if output_shape[axis] is different from input_shape[axis]
        # we must resize it again. We do it with order = 0
        if np.any(output_shape!=tmp_img.shape):
            do_additional_resize = True
            order = 0
            img = tmp_img
        else:
            new_img = tmp_img
    
    # normal resizing
    if not do_anisotropy or do_additional_resize:
        new_img = np.empty(output_shape)
        for c in range(input_shape[0]):
            new_img[c] = resize_fct(img[c], output_shape[1:], order=order, **resize_kwargs)
            
    return new_img

#---------------------------------------------------------------------------
# Image reader

def tif_read_meta(tif_path, display=False):
    """
    read the metadata of a tif file and stores them in a python dict.
    if there is a 'ImageDescription' tag, it transforms it as a dictionary
    """
    meta = {}
    with tiff.TiffFile(tif_path) as tif:
        for page in tif.pages:
            for tag in page.tags:
                tag_name, tag_value = tag.name, tag.value
                if display: print(tag.name, tag.code, tag.dtype, tag.count, tag.value)

                # below; fix storage problem for ImageDescription tag
                if tag_name == 'ImageDescription': 
                    list_desc = tag_value.split('\n')
                    dict_desc = {}
                    for idx, elm in enumerate(list_desc):
                        split = elm.split('=')
                        dict_desc[split[0]] = split[1]
                    meta[tag_name] = dict_desc
                else:
                    meta[tag_name] = tag_value
            break # just check the first image
    return meta

def tif_get_spacing(path, res=1e-6):
    """
    get the image spacing stored in the metadata file.
    """
    img_meta = tif_read_meta(path)

    xres = (img_meta["XResolution"][1]/img_meta["XResolution"][0])*res
    yres = (img_meta["YResolution"][1]/img_meta["YResolution"][0])*res
    zres = float(img_meta["ImageDescription"]["spacing"])*res

    return (xres, yres, zres)

def tif_read_imagej(img_path):
    """Read tif file metadata stored in a ImageJ format.
    adapted from: https://forum.image.sc/t/python-copy-all-metadata-from-one-multipage-tif-to-another/26597/8

    Parameters
    ----------
    img_path : str
        Path to the input image.

    Returns
    -------
    img : numpy.ndarray
        Image.
    img_meta : dict
        Image metadata. 
    """

    with tiff.TiffFile(img_path) as tif:
        assert tif.is_imagej

        # store img_meta
        img_meta = {}

        # get image resolution from TIFF tags
        tags = tif.pages[0].tags
        x_resolution = tags['XResolution'].value
        y_resolution = tags['YResolution'].value
        resolution_unit = tags['ResolutionUnit'].value
        
        img_meta["resolution"] = (x_resolution, y_resolution, resolution_unit)

        # parse ImageJ metadata from the ImageDescription tag
        ij_description = tags['ImageDescription'].value
        ij_description_metadata = tiff.tifffile.imagej_description_metadata(ij_description)
        # remove conflicting entries from the ImageJ metadata
        ij_description_metadata = {k: v for k, v in ij_description_metadata.items()
                                   if k not in 'ImageJ images channels slices frames'}

        img_meta["description"] = ij_description_metadata
        
        # compute spacing
        xres = (x_resolution[1]/x_resolution[0])
        yres = (y_resolution[1]/y_resolution[0])
        zres = float(ij_description_metadata["spacing"])
        
        img_meta["spacing"] = (xres, yres, zres)

        # read the whole image stack and get the axes order
        series = tif.series[0]
        img = series.asarray()
        
        img_meta["axes"] = series.axes
    
    return img, img_meta

def imread(img_path):
    """
    use skimage imread or sitk imread depending on the file extension:
    .tif --> skimage.io.imread
    .nii.gz --> SimpleITK.imread
    """
    extension = img_path[img_path.rfind('.'):].lower()
    if extension == ".tif" or extension == ".tiff":
        try: 
            img, img_meta = tif_read_imagej(img_path)  # try loading ImageJ metadata for tif files
            return img, img_meta
        except:   
            img_meta = {}    
            try: img_meta["spacing"] = tif_get_spacing(img_path)
            except: img_meta["spacing"] = []
    
            return io.imread(img_path), img_meta 
    else:
        print("[Error] Unknown image format:", extension)

#---------------------------------------------------------------------------
# Utils for volume meshing

def mesh3d(verts, img, remove_label=0):
    """Creates a 3D mesh from a list of vertices. Uses the image as a mask to remove unwanted triangles.
    Returns list of triangles.
    https://forum.image.sc/t/create-3d-volume-mesh/34052/9
    """
    # Delaunay
    tri = Delaunay(verts).simplices

    # filter inner triangles only
    center = verts[tri[:,0]] + verts[tri[:,1]] + verts[tri[:,2]] + verts[tri[:,3]]
    msk = img[tuple((center/4).astype(int).T)] != remove_label
    tri_sorted = tri[msk]
    return tri_sorted

def tetramesh_vol(a, b, c, d):
    """Volume of a tetrahedron mesh. a, b, c, d are the tetrahedron vertices with 3d coodinates.
    shape of a, b, c, or d are either (3,) or (N,3) where N is the number of vertices in the mesh.
    https://en.wikipedia.org/wiki/Tetrahedron#General_properties
    """
    denom = ((a-d)*(np.cross((b-d),(c-d))))
    denom = denom.sum(axis = 1 if len(denom.shape) > 1 else 0)
    return abs(denom).sum()/6

def compute_sphericity(volume, surface):
    """Return sphericity.
    https://en.wikipedia.org/wiki/Sphericity
    """
    return (pow(np.pi,1/3)*pow(6*volume,2/3))/surface

#---------------------------------------------------------------------------
# Compute volume, surface and mesh

def get_resample_shape(input_shape, spacing):
    """Debugging function.
    """
    input_shape = np.array(input_shape)
    spacing = np.array(spacing)
    if len(input_shape)==4:
        input_shape=input_shape[1:]
    return np.round(((spacing/spacing.min())[::-1]*input_shape)).astype(int)

def safe_imread(img_path, spacing=()):
    # read image
    assert pathlib.Path(img_path).suffix == '.tif', "[Error] Current version of compute params only works with tif file but attempted to read: {}".format(img_path)
    img, metadata = imread(img_path)
    if len(metadata['spacing'])==3 and len(spacing)==0: spacing = np.array(metadata['spacing'])

    assert len(img.shape)==3 or (len(img.shape)==4 and img.shape[0]==1), "[Error] Strange image shape ({}). Please provide a 3d image".format(img.shape)
    
    # sanity check: only 0 or 1 label are allowed
    unq, counts = np.unique(img, return_counts=True)
    if len(unq)!=2:
        print("[Warning] Only 2 class annotations are allowed (0 or 1) but found {}. A threshold will be applied but might cause some issues.".format(unq))
        # set background voxels to 0 and foreground to 1
        img = (img != unq[np.argmax(counts)]).astype(np.uint8)

    # warning if no spacing
    if len(spacing)==0:
        print("[Warning] No spacing has been defined. The result will be expressed in voxel units.")
    # BELOW: for debugging
    # else:
    #     output_shape = get_resample_shape(img.shape, spacing)
    #     img = resize_3d(np.expand_dims(img,0), output_shape, is_msk=True, order=1)[0]
    # return img, tuple(spacing.min() for _ in spacing)
    
    return img, spacing

def compute_volume_surface_sphericity(img, bg=None, spacing=(), verbose=False):
    """Compute volume, surface and sphericity of a volumetric object.

    Parameters
    ----------
    img : numpy.ndarray
        Image array.
    bg : int, default=None
        Value of the background voxels. If bg is None then use the most frequent value.
    spacing : tuple, default=()
        Image spacing.
    verbose : boolean, default=False
        Whether to display information.
    """
    if bg is None:
        # compute volume with voxel
        labels = measure.label(img)
        unq,vol_voxel = np.unique(labels, return_counts=True)

        if len(unq)>2:
            print("[Warning] More than one object were found in the image. Number of connected components: {}".format(len(unq)-1))

        if verbose: print("Voxel volume:", vol_voxel[1:])

        # use the biggest volume as background
        bg = unq[np.argmax(vol_voxel)]
    
    # Marching cube
    verts, faces, normals, values = measure.marching_cubes(img, 0.5) 
    
    # create and sort correct tetrahedron to obtain a volume mesh
    tetra = mesh3d(verts=verts, img=img, remove_label=bg)
    
    # compute the volume
    volume = tetramesh_vol(verts[tetra[:,0]],verts[tetra[:,1]],verts[tetra[:,2]],verts[tetra[:,3]])
    
    # compute the surface
    surface = measure.mesh_surface_area(verts=verts, faces=faces)
    
    # compute the sphericity
    sphericity = compute_sphericity(volume=volume, surface=surface)
    
    # eventually adapt to spacing
    if len(spacing)>0:
        volume = volume*np.prod(spacing)
        surface = surface*np.prod(spacing)
    
    # display for debugging
    if verbose:
        labels = measure.label(img, background=bg)
        unq,vol_voxel = np.unique(labels, return_counts=True)
        vol_voxel = np.sum(vol_voxel[1:])
        if len(spacing)>0: vol_voxel=vol_voxel*np.prod(spacing)
        print("Number of labels:", unq)
        print("Volume (voxel):", vol_voxel)
        print("Volume (mesh):", volume)
        print("Surface:", surface)
        print("Sphericity:", sphericity)
    
    return volume, surface, sphericity

def compute_flatness_elongation(img, bg=None, spacing=(), verbose=False):
    """Compute flatness and elongation of a volumetric object.
    These are called "Shape factor": https://en.wikipedia.org/wiki/Shape_factor_%28image_analysis_and_microscopy%29#Elongation_shape_factor

    Parameters
    ----------
    img : numpy.ndarray
        Image array.
    bg : int, default=None
        Value of the background voxels. If bg is None then use the most frequent value.
    spacing : tuple, default=()
        Image spacing.
    verbose : boolean, default=False
        Whether to display information.
    """
    if bg is None:
        # compute volume with voxel
        labels = measure.label(img)
        unq,vol_voxel = np.unique(labels, return_counts=True)

        if len(unq)>2:
            print("[Warning] More than one object were found in the image. Number of connected components: {}".format(len(unq)-1))

        if verbose: print("Voxel volume:", vol_voxel[1:])

        # use the biggest volume as background
        bg = unq[np.argmax(vol_voxel)]

    # get foreground voxel coordinates
    fg = np.argwhere(img != bg)

    # compute barycenter and center the foreground voxels
    bary = np.mean(fg,axis=0)
    fg = fg - bary
    # fg = fg / np.sqrt(np.mean(fg*fg))

    # get the covariance matrix
    cov = fg.T.dot(fg)/len(fg)

    # other method:
    # m = measure.moments_central(img, order=2)
    # cov = np.array([
    #     [m[2,0,0], m[1,1,0], m[1,0,1]],
    #     [m[1,1,0], m[0,2,0], m[0,1,1]],
    #     [m[1,0,1], m[0,1,1], m[0,0,2]],
    # ])/m[0,0,0]

    # eventually resize image
    if len(spacing)>0:
        spacing = np.array(spacing)[::-1].reshape(1,3)
        cov = cov*(spacing.T.dot(spacing))

    # get the eigenvalues
    eigval = np.real(sorted(np.linalg.eigvals(cov)))

    # compute flatness and elongation
    flatness = np.sqrt(eigval[1]/eigval[0])
    elongation = np.sqrt(eigval[2]/eigval[1])

    return flatness, elongation

def compute_number_vmean_vtot(img, cc_img, bg=None, spacing=(), min_vol=2, verbose=False):
    """Compute for an nucleus image the number of chromocentres located in the nucleus mask as well as the mean volume and the total volume of the chromocentres. By default, the background voxels are automatical set to the class with the largest number of voxels. `min_vol` is the minimal number of voxels in a connected component region to be considered a valid chromocentre, smaller or equal regions are not considered. 
    """
    if bg is None:
        # compute volume with voxel
        labels = measure.label(img)
        unq,vol_voxel = np.unique(labels, return_counts=True)

        if len(unq)>2:
            print("[Warning] More than one object were found in the image. Number of connected components: {}".format(len(unq)-1))

        if verbose: print("Voxel volume:", vol_voxel[1:])

        # use the biggest volume as background
        bg = unq[np.argmax(vol_voxel)]
    
    # select only chromocenters in the nucleus
    # CAREFUL: the connectivity is set to 1!
    labels = measure.label(np.logical_and(img!=bg,cc_img!=bg).astype(int), background=bg, connectivity=1)
    
    # compute volumes
    unq,vol_voxel = np.unique(labels, return_counts=True)

    # remove too small volumes
    # if not np.all(vol_voxel>min_vol):
    unq, vol_voxel = unq[vol_voxel>min_vol], vol_voxel[vol_voxel>min_vol]

    # number_vmean_vtot
    number = len(unq)-1
    vmean = np.mean(vol_voxel[unq!=bg])
    vtot = np.sum(vol_voxel[unq!=bg])

    # No need because images are already resampled!
    # set the spacing if needed
    if len(spacing)>0:
        vmean = vmean*np.prod(spacing)
        vtot = vtot*np.prod(spacing)

    return number, vmean, vtot

    
class ComputeParams:
    """Compute Nucleus and Chromocenter parameters.

    Parameters
    ----------
    nc_path : str
        Path to the nucleus image.
    bg : int, default=0
        Value of the background voxels.
    cc_path : str, default=None
        Path to a chromocentre image.
    cc_min_vol : int, default=2
        Minimal number of voxel in chromocentre to be considered valid. Smaller or equal volume are not considered.
    spacing : tuple, default=()
        Image spacing.
    verbose : boolean, default=False
        Whether to display information.
    """
    NUCLEUS_KEYS = [
        'Volume',
        'SurfaceArea',
        'Sphericity',
        'Flatness',
        'Elongation',
    ]

    CHROMOCENTER_KEYS = [
        'NbCc',
        'VCcMean',
        'VCcTotal',
        'VolumeRHF',
    ]

    def __init__(self, nc_path, cc_path=None, bg=0, spacing=(), cc_min_vol=2, verbose=False):
        # stores nucleus and chromocenter parameters
        self.nc_params = {}
        self.cc_params = {}

        # path and spacing
        self.nc_path = nc_path
        self.spacing  = np.array(spacing, dtype=np.float64)

        # information
        if verbose: print("Background voxel is set to", bg)

        # read nucleus image and metadata
        self.nc_imag, self.spacing = safe_imread(img_path=self.nc_path, spacing=self.spacing)

        if verbose: print("Spacing:", self.spacing)

        # read chromocenter image and metadata
        if cc_path is not None:
            self.cc_imag, _ = safe_imread(img_path=cc_path, spacing=self.spacing)

        # nucleus volume, surface and sphericity computation
        self.nc_params['Volume'], self.nc_params['SurfaceArea'], self.nc_params['Sphericity'] = compute_volume_surface_sphericity(self.nc_imag, bg=bg, spacing=self.spacing, verbose=verbose)

        # nucleus flatness and elongation
        self.nc_params['Flatness'], self.nc_params['Elongation'] = compute_flatness_elongation(self.nc_imag, bg=bg, spacing=self.spacing, verbose=verbose)

        # chromocenters computation
        if cc_path is not None:
            self.cc_params['NbCc'], self.cc_params['VCcMean'], self.cc_params['VCcTotal'] = compute_number_vmean_vtot(img=self.nc_imag, cc_img=self.cc_imag, bg=bg, spacing=self.spacing, min_vol=cc_min_vol, verbose=verbose)

            # add VolumeRHF
            self.cc_params['VolumeRHF'] = self.cc_params['VCcTotal']/self.nc_params['Volume']
        else:
            self.cc_params['NbCc'], self.cc_params['VCcMean'], self.cc_params['VCcTotal'] = 0, 0, 0
            self.cc_params['VolumeRHF'] = np.inf

        
    
    def __str__(self):
        out = "filename: {}\n".format(self.nc_path)
        out += "".join("{}: {}\n".format(k, v) for k,v in self.nc_params.items()) # nucleus
        out += "".join("{}: {}\n".format(k, v) for k,v in self.cc_params.items()) # chromocenter
        return out

def find_leaf_directory(dic):
    """Find the leaf directory in a nested directories. There should be only one directory at each level.
    """
    root, child_dic = next(os.walk(dic))[:2]
    child_dic = [os.path.join(root, cd) for cd in child_dic]
    assert len(child_dic)<=1, "[Error] Found two or more nested directories in {}: {}".format(dic, child_dic)
    if len(child_dic)==1:
        return find_leaf_directory(child_dic[0])
    else:
        return dic
        

def compute_directory(path, cc_path=None, bg=0, spacing=(), out_path="params", cc_min_vol=2, verbose=False):
    """Same as compute_volume_surface_sphericity but on a directory. Output results in a csv file.
    """
    if len(spacing)>0:
        spacing = np.array(spacing, dtype=np.float64)

    path = find_leaf_directory(path)
    if cc_path is not None: cc_path = find_leaf_directory(cc_path)
    filenames = os.listdir(path)
    out_params = {'NucleusFileName': filenames}
    for k in ComputeParams.NUCLEUS_KEYS: out_params[k]=[]
    for k in ComputeParams.CHROMOCENTER_KEYS: out_params[k]=[]

    for i in range(len(filenames)):
        print("[{}/{}] Computing parameters for {}".format(i,len(filenames),filenames[i]))
        img_path = os.path.join(path, filenames[i])
        if cc_path is not None: 
            cc_img_path = os.path.join(cc_path, filenames[i])
            if not os.path.exists(cc_img_path): cc_img_path = None
        else: cc_img_path = None

         # compute parameters for that image
        comp_params = ComputeParams(
            nc_path=img_path,
            cc_path=cc_img_path,
            bg=bg,
            spacing=spacing,
            cc_min_vol=cc_min_vol,
            verbose=verbose)

        # store nucleus parameters in the output dictionary
        for k,v in comp_params.nc_params.items():
            out_params[k] += [v]
        
        # store chromocenter parameters
        for k,v in comp_params.cc_params.items():
            out_params[k] += [v]

    df = pd.DataFrame(out_params)
    df.to_csv(out_path+'.csv', index=False)
    # df.to_excel("params.xlsx", index=False)

#---------------------------------------------------------------------------
# Argparse

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Compute objects characteristics.")
    parser.add_argument("-p", "--path", type=str, 
        help="Path to an image or a folder of images.")
    parser.add_argument("-cc", "--chromo", type=str, default=None,
        help="Path to a chromocenter image or a folder of chromocenter images.")
    parser.add_argument("-s", "--spacing", type=str, nargs='+', default=(),
        help="Image spacing. Example: 0.1032 0.1032 0.2")
    parser.add_argument("-o", "--out_path", type=str, default="params",
        help="(default=params) Path to the output CSV file.")
    parser.add_argument("-b", "--bg_value", type=int, default=0,
        help="(default=0) Value of the background voxels.")
    parser.add_argument("-cv", "--cc_min_vol", type=int, default=2,
        help="(default=2) Minimal number of voxel in chromocentre to be considered valid. Smaller or equal volume are not considered.")
    parser.add_argument("-v", "--verbose", default=False,  action='store_true', dest='verbose',
        help="Display some information.") 
    args = parser.parse_args()
    
    if os.path.isdir(args.path):
        compute_directory(
            path=args.path,
            cc_path=args.chromo,
            bg=args.bg_value,
            spacing=args.spacing,
            out_path=args.out_path,
            cc_min_vol=args.cc_min_vol,
            verbose=args.verbose)
    else:
        params = ComputeParams(
            nc_path=args.path,
            cc_path=args.chromo,
            spacing=args.spacing,
            cc_min_vol=args.cc_min_vol,
            verbose=args.verbose)
        print(params)

#python3 compute_params.py -p "/home/stagiaire-pt/Willy-wanka/Aline_images/graham_sortie/" -cc "/home/stagiaire-pt/Willy-wanka/Aline_images/raw_C0_ACQUISITIONS_3D_Chromocenter_Predictions/20230915-004004-chromo_aline_all_fold0" -s 0.1032 0.1032 0.2 n
