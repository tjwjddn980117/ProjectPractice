import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from ..miscc.config import cfg

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
    ''' collect all image files what we have. like \'.jpg\', \'.PNG\', ... '''
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_imgs(img_path, imsize, bbox=None, transform=None, normalize=None):
    ''' 
    This function loads a given image, optionally cuts out a specific area, 
    applies a transformation, readjusts to multiple sizes, 
    and returns the normalized results to the list.

    Arguments:
        img_path (str) : the path of image dir.
        imsize ( ) : it should width, height.
        bbox (list) : boundary box. standard for cutting out a picture.
        transform ( ) : the parameter that standard transform.
        normalize ( ) : the parameter that standard normalize.
    
    Returns:
        list [Image] : list of resized images data type is (Image).
    '''
    img = Image.open(img_path).convert('RGB')
    width, height = img.size

    # cropping the image
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75) # canculate maximum range
        center_x = int((2 * bbox[0] + bbox[2]) / 2) # define center_x
        center_y = int((2 * bbox[1] + bbox[3]) / 2) # define center_y
        y1 = np.maximum(0, center_y - r) # define y1's point
        y2 = np.minimum(height, center_y + r) # define y2's point
        x1 = np.maximum(0, center_x - r) # define x1's point
        x2 = np.minimum(width, center_x + r) # define x2's point
        img = img.crop([x1, y1, x2, y2]) # cropping
        # this function is just cropping with pictures

    if transform is not None:
        img = transform(img)
    
    ret = []
    for i in range(cfg.TREE.BRANCH_NUM):
        # resizing image for batch_size
        if i < (cfg.TREE.BRANCH_NUM - 1):
            re_img = transforms.Scale(imsize[i])(img)
        else:
            # last image dosen't transform scale last batch
            re_img = img
        
        # append all re_sizing image with normalize
        ret.append(normalize(re_img))
    
    return ret