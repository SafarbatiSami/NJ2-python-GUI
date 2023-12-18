import argparse
import subprocess
import sys
import os

def install(packages):
    """Install a list of packages.
    """
    for p in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", p])

try:
    import tifffile as tiff
    from skimage import io
    import numpy as np

except ImportError as e:
    print("[Warning] Some packages are missing. Installing...")
    install(['tifffile', 'scikit-image', 'numpy'])
    import tifffile as tiff
    from skimage import io
    import numpy as np


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


def convert_tif_imagej(path, out_path=None, spacing=(), res=1e-6, unit='MICROMETER'):
    """Convert one TIFF image to ImageJ format.
    """
    spacing = np.array(spacing)

    img, metadata = imread(path)

    img = (img * 255 / img.max()).astype(int)

    img = tiff.transpose_axes(img, 'ZYX', 'TZCYXS')

    if out_path is None: out_path = path

    if len(spacing)>0 and len(metadata['spacing'])==0:
        resolution = ((int(1/res), int(spacing[0]/res)),(int(1/res), int(spacing[1]/res)),unit)
        metadata = {'spacing': spacing[2], 'unit':unit}
    elif len(metadata['spacing'])>0:
        spacing = np.array(metadata['spacing'])
        resolution = ((int(1/res), int(spacing[0]/res)),(int(1/res), int(spacing[1]/res)),unit)
        metadata = {'spacing': spacing[2], 'unit':unit}
    else:
        resolution = None
        metadata = None 

    tiff.imwrite(out_path, img.astype(np.uint8),
        resolution=resolution,
        imagej=True, 
        metadata=metadata,
        compression=('zlib', 1))
    
def convert_directory(path, out_path=None, spacing=(), res=1e-6):
    """Same as compute_volume_surface_sphericity but on a directory. Output results in a csv file.
    """
    if len(spacing)>0:
        spacing = np.array(spacing, dtype=np.float64)

    filenames = os.listdir(path)

    if out_path is None: out_path = path

    for i in range(len(filenames)):
        print("[{}/{}] Converting {}".format(i,len(filenames),filenames[i]))
        img_path = os.path.join(path, filenames[i])
        img_out_path = os.path.join(out_path, filenames[i])

        convert_tif_imagej(img_path, img_out_path, spacing=spacing, res=res)

        

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Convert TIFF files to ImageJ format.")
    parser.add_argument("-p", "--path", type=str, 
        help="Path to an image or a folder of images.")
    parser.add_argument("-o", "--out_path", type=str, default=None,
        help="Path to a folder to store the converted image(s).")
    parser.add_argument("-s", "--spacing", type=str, nargs='+', default=(),
        help="Image spacing. Example: 0.1032 0.1032 0.2")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        convert_directory(path=args.path, out_path=args.out_path, spacing=args.spacing, res=1e-6)
    else:
        convert_tif_imagej(path=args.path, out_path=args.out_path, spacing=args.spacing, res=1e-6)


