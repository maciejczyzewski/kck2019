print("[IMAGES]")
# montage ready_*.jpg -geometry +2+2 samoloty.jpg

import random
import numpy as np

from glob import glob
from tqdm import tqdm
from skimage import io
from skimage import img_as_float
from scipy import ndimage as ndi
from skimage.feature import canny
from skimage.color import rgb2hed
from skimage.transform import resize
from skimage.measure import regionprops
from skimage.exposure import rescale_intensity
from skimage.restoration import denoise_nl_means
from skimage.filters import prewitt, threshold_minimum
from skimage.morphology import reconstruction, remove_small_objects
from skimage.color import rgb2gray, label2rgb
from PIL import Image, ImageDraw

for path in tqdm(glob("data/planes/*.jpg")):
    print(path)
    img = io.imread(path)
    img_resized = resize(img, (300, int(img.shape[1] * 300 / img.shape[0])),
                         anti_aliasing=True)

    image = img_as_float(img_resized)

    ihc_hed = rgb2hed(image)
    h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1))
    d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1))
    zdh = np.dstack((np.zeros_like(h), d, h))

    image = rgb2gray(zdh)

    p2, p98 = np.percentile(image, (0.5, 99.5))
    image = rescale_intensity(image, in_range=(p2, p98))

    image = prewitt(image)

    patch_kw = dict(
        patch_size=1,  # 5x5 patches
        patch_distance=5,  # 13x13 search area
        multichannel=False,
    )

    image = denoise_nl_means(image, h=0.95, fast_mode=True, **patch_kw)

    dilated = image

    dilated_black = dilated

    thresh = threshold_minimum(dilated_black)
    binary = dilated_black > thresh
    dilated_black = binary

    elevation_map = prewitt(dilated_black)

    edges = canny(elevation_map)

    fill_planes = ndi.binary_fill_holes(edges)
    fill_planes = remove_small_objects(fill_planes, 10 * 10)

    labeled_planes, _ = ndi.label(fill_planes)
    # image_label_overlay = label2rgb(labeled_planes, image=img_resized)

    regions = regionprops(labeled_planes)

    image = Image.fromarray((img_resized * 255).astype(np.uint8))

    draw = ImageDraw.Draw(image)

    color_but_not_blue = lambda: (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255 // 2),
    )
    colors = ["#%02x%02x%02x" % color_but_not_blue() for _ in range(20)]
    colors_idx = 0
    for props in regions:
        y0, x0 = props.centroid
        arr = [
            (x0 - 1, y0 - 1),
            (x0 - 1, y0),
            (x0 - 1, y0 + 1),
            (x0, y0 - 1),
            (x0, y0),
            (x0, y0 + 1),
            (x0 + 1, y0 - 1),
            (x0 + 1, y0),
            (x0 + 1, y0 + 1),
        ]

        draw.point(arr)
        coords = []
        fimg = props.filled_image
        nimg = np.zeros(fimg.shape)
        print("+ samolot")
        fimg = np.pad(fimg, [(0, 1), (0, 1)], mode="constant")
        for i in range(fimg.shape[0]):
            for j in range(fimg.shape[1]):
                try:
                    if fimg[i, j] is False:
                        continue
                    arr = [
                        fimg[i - 1, j - 1],
                        fimg[i - 1, j],
                        fimg[i - 1, j + 1],
                        fimg[i, j - 1],
                        fimg[i, j + 1],
                        fimg[i + 1, j - 1],
                        fimg[i + 1, j],
                        fimg[i + 1, j + 1],
                    ]
                    if int(fimg[i, j]) is not 0 and 0 in map(int, arr):
                        nimg[i, j] = 1
                except:
                    pass
        indx = np.nonzero(nimg)
        coords = np.vstack(
            [indx[i] + props.slice[i].start for i in range(props._ndim)]).T
        coords_norm = [(pts[1], pts[0]) for pts in coords]
        draw.point(coords_norm, fill=colors[colors_idx])
        colors_idx += 1

    # image.show()
    image.save(f"ready_{path.split('/')[-1]}")
