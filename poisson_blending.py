import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy import signal
from scipy.sparse.linalg import spsolve
from scipy.ndimage.morphology import binary_dilation
import argparse

LAPLACIAN_FILTER = np.array([[0,  1,  0],
                             [1, -4,  1],
                             [0,  1,  0]])

def create_laplacian(h, w, inside_vec):
    N = h*w
    main_diag = (1 - inside_vec) -4 * inside_vec # 1 if outside and -4 if inside

    right_side_diag = np.ones(N-1) * inside_vec[:-1] # 0 if outside and 1 if inside
    right_side_diag[np.arange(1,N)%w==0] = 0 # handle the end of the row
    left_side_diag = np.ones(N-1) * inside_vec[1:] # 0 if outside and 1 if inside
    left_side_diag[np.arange(N-1)%w==0] = 0 # handle the end of the row
    up_diag = np.ones(N-w) * inside_vec[w:]
    down_diag = np.ones(N-w) * inside_vec[:-w]
    diagonals = [main_diag,left_side_diag,right_side_diag,down_diag, up_diag]
    laplacian = scipy.sparse.diags(diagonals, [0, -1, 1, w, -w], format="csr")

    return laplacian

def crop_src_image(im_src, mask):
    Ys,Xs = np.where(mask != 0 )
    return im_src[min(Ys): max(Ys)+1, min(Xs): max(Xs)+1, :],\
           mask[min(Ys): max(Ys)+1, min(Xs): max(Xs)+1]

def get_src_and_tgt_ranges(center, src_shape, tgt_shape):
    h_src, w_src, _ = src_shape
    h_tgt, w_tgt, _ = tgt_shape
    c_x, c_y= (center[0], center[1])

    tgt_min_y = c_y-h_src//2
    tgt_max_y = c_y+h_src//2 if h_src % 2 == 0 else c_y+h_src//2 + 1
    tgt_min_x = c_x-w_src//2
    tgt_max_x = c_x+w_src//2 if w_src % 2 == 0 else c_x+w_src//2 + 1

    src_min_y = -tgt_min_y if (tgt_min_y < 0) else 0
    src_max_y = h_src-(tgt_max_y-h_tgt) if (tgt_max_y > h_tgt) else h_src
    src_min_x = -tgt_min_x if (tgt_min_x < 0) else 0
    src_max_x = w_src-(tgt_max_x-w_tgt) if (tgt_max_x > w_tgt) else w_src

    return {
        "tgt_min_y": tgt_min_y,
        "tgt_max_y": tgt_max_y,
        "tgt_min_x": tgt_min_x,
        "tgt_max_x": tgt_max_x,
        "src_min_y": src_min_y,
        "src_max_y": src_max_y,
        "src_min_x": src_min_x,
        "src_max_x": src_max_x
    }

def get_border_mask(mask):
    k = np.ones((3,3),dtype=int) # for 4-connected
    mask_border = binary_dilation(mask==0, k).astype(mask.dtype) & mask
    return mask_border

def poisson_blend(im_src, im_tgt, im_mask, center):
    im_src = im_src.astype(np.float64)
    im_tgt = im_tgt.astype(np.float64)

    im_src, im_mask = crop_src_image(im_src, im_mask)

    assert im_mask.shape[0] <= im_tgt.shape[0] and im_mask.shape[1] <= im_tgt.shape[1]

    mask = (im_mask > 0).astype(im_mask.dtype)
    mask_border = get_border_mask(mask)
    ranges = get_src_and_tgt_ranges(center, im_src.shape, im_tgt.shape)
    blend_out = np.zeros_like(im_tgt)
    
    # Set pixels from the target image that are outside source ovelap
    blend_out[:ranges["tgt_min_y"], :, :] = im_tgt[:ranges["tgt_min_y"], :, :]
    blend_out[ranges["tgt_max_y"]:, :, :] = im_tgt[ranges["tgt_max_y"]:, :, :]
    blend_out[:, :ranges["tgt_min_x"], :] = im_tgt[:, :ranges["tgt_min_x"], :]
    blend_out[:, ranges["tgt_max_x"]:, :] = im_tgt[:, ranges["tgt_max_x"]:, :]
    
    for channel in range(im_tgt.shape[2]):
        src_lap = signal.convolve2d(im_src[:,:, channel], LAPLACIAN_FILTER, mode='same')

        # Set pixels from the source/target image that are inside the source overlap
        src_patch_in_tgt = im_tgt[ranges["tgt_min_y"]:ranges["tgt_max_y"], ranges["tgt_min_x"]:ranges["tgt_max_x"], channel]
        src_patch_in_im_blend = blend_out[ranges["tgt_min_y"]:ranges["tgt_max_y"], ranges["tgt_min_x"]:ranges["tgt_max_x"], channel]
        src_patch_src_lap = src_lap[ranges["src_min_y"]:ranges["src_max_y"], ranges["src_min_x"]:ranges["src_max_x"]]
        src_patch_border_mask = mask_border[ranges["src_min_y"]:ranges["src_max_y"], ranges["src_min_x"]:ranges["src_max_x"]]
        src_patch_mask = mask[ranges["src_min_y"]:ranges["src_max_y"], ranges["src_min_x"]:ranges["src_max_x"]]

        h_tgt, w_tgt = src_patch_in_tgt.shape

        # building vectors for A and b
        im_vec = src_patch_in_im_blend.flatten()
        tgt_vec = src_patch_in_tgt.flatten()
        src_lap_vec = src_patch_src_lap.flatten()
        border_mask_vec = src_patch_border_mask.flatten()
        mask_vec = src_patch_mask.flatten()

        # building b and A
        b = np.zeros_like(im_vec)
        inside_vec = np.logical_and(mask_vec == 1, border_mask_vec == 0)
        b[inside_vec == False] = tgt_vec[inside_vec == False]
        b[inside_vec == True] = src_lap_vec[inside_vec == True]

        A = create_laplacian(h_tgt, w_tgt, inside_vec)

        im_vec = spsolve(A, b)
        blend_out[ranges["tgt_min_y"]:ranges["tgt_max_y"], ranges["tgt_min_x"]:ranges["tgt_max_x"], channel] = im_vec.reshape((h_tgt, w_tgt))

    return blend_out


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana2.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./our_masks/banana2.jpg', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='mask file path')
    parser.add_argument('--out_path', type=str, default='im_blend.png', help='mask file path')
    return parser.parse_args()

if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    # im_tgt = cv2.resize(im_tgt, (0,0), fx=0.125, fy=0.125)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    # im_src = cv2.resize(im_src, (0,0), fx=0.125, fy=0.125)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        # im_mask = cv2.resize(im_mask, (0,0), fx=0.125, fy=0.125)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imwrite(args.out_path, im_clone)
    # cv2.imshow('Cloned image', im_clone)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
