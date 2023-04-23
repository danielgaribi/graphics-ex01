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

PYRAMID_FILTER = 1.0 / 256 * np.array([[1, 4, 6, 4, 1],
                                       [4, 16, 24, 16, 4],
                                       [6, 24, 36, 24, 6],
                                       [4, 16, 24, 16, 4],
                                       [1, 4, 6, 4, 1]])

def build_pyramid(image: np.ndarray, num_levels: int) -> list[np.ndarray]:
    """Coverts image to a pyramid list of size num_levels.
    """
    P = [image.copy()]
    L = []
    
    for i in range(num_levels):
        conv = signal.convolve2d(P[-1], PYRAMID_FILTER, mode='same', boundary='symm')
        L.append(P[-1] - conv)
        decimated = conv[::2, ::2] 

        # current = cv2.pyrDown(P[-1])
        # current_up = cv2.resize(current, (P[-1].shape[1], P[-1].shape[0]))
        # L.append(P[-1] - current_up)
        P.append(decimated)
    
    L.append(P[-1])
        
    return P, L

def create_laplacian(h, w, inside_vec):
    N = h*w
    main_diag = (1 - inside_vec) -4 * inside_vec # 1 if outside and -4 if inside

    right_side_diag = np.ones(N-1) * inside_vec[:-1] # 0 if outside and 1 if inside
    right_side_diag[np.arange(1,N)%w==0] = 0 # handel the end of the row
    left_side_diag = np.ones(N-1) * inside_vec[1:] # 0 if outside and 1 if inside
    left_side_diag[np.arange(1,N)%w==0] = 0 # handel the end of the row
    up_down_diag = np.ones(N-w) * inside_vec[:-w]
    diagonals = [main_diag,left_side_diag,right_side_diag,up_down_diag,up_down_diag]
    laplacian = scipy.sparse.diags(diagonals, [0, -1, 1, w, -w], format="csr")

    return laplacian

def get_src_and_tgt_ranges(center, level, src_shape, tgt_shape):
    h_src, w_src = src_shape
    h_tgt, w_tgt = tgt_shape
    c_y, c_x = (center[0] // (2**level), center[1] // (2**level))

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

def create_target_mask(src_mask, center, level, tgt_shape):
    ranges = get_src_and_tgt_ranges(center, level, src_mask.shape, tgt_shape)
    # h_src, w_src = src_mask.shape
    # h_tgt, w_tgt = tgt_shape
    # c_y, c_x = (center[0] // (2**level), center[1] // (2**level))

    # min_y = c_y-h_src//2
    # max_y = c_y+h_src//2 if h_src % 2 == 0 else c_y+h_src//2 + 1

    # min_x = c_x-w_src//2
    # max_x = c_x+w_src//2 if w_src % 2 == 0 else c_x+w_src//2 + 1

    # mask_min_y = -min_y if (min_y < 0) else 0
    # mask_max_y = h_src-(max_y-h_tgt) if (max_y > h_tgt) else h_src
    # mask_min_x = -min_x if (min_x < 0) else 0
    # mask_max_x = w_src-(max_x-w_tgt) if (max_x > w_tgt) else w_src


    mask_tgt = np.zeros(tgt_shape)
    mask_tgt[ranges.tgt_min_y:ranges.tgt_max_y, ranges.tgt_min_x:ranges.tgt_max_x] = src_mask[ranges.src_min_y:ranges.src_max_y, ranges.src_min_x:ranges.src_max_x]

    k = np.ones((3,3),dtype=int) # for 4-connected
    mask_border = binary_dilation(mask_tgt, k)
    return mask_tgt, mask_border

def get_border_mask(mask):
    k = np.ones((3,3),dtype=int) # for 4-connected
    mask_border = binary_dilation(mask==0, k).astype(mask.dtype) & mask
    return mask_border

def poisson_blend(im_src, im_tgt, im_mask, center):
    # TODO: Implement Poisson blending of the source image onto the target ROI
    assert im_mask.shape[0] <= im_tgt.shape[0] and im_mask.shape[1] <= im_tgt.shape[1]

    NUM_LEVELS = 5

    im_src = cv2.cvtColor(im_src, cv2.COLOR_BGR2GRAY)
    im_tgt = cv2.cvtColor(im_tgt, cv2.COLOR_BGR2GRAY)
    plt.imshow(im_src, cmap='gray')
    plt.savefig("im_src.png")
    plt.close()
    plt.imshow(im_tgt, cmap='gray')
    plt.savefig("im_tgt.png")
    plt.close()

    P_src_pyramid, L_src_pyramid = build_pyramid(im_src, NUM_LEVELS)
    P_tgt_pyramid, L_tgt_pyramid = build_pyramid(im_tgt, NUM_LEVELS)
    mask_pyramid, _ = build_pyramid(im_mask, NUM_LEVELS)
    mask_pyramid = [(mask > 0).astype(im_mask.dtype) for mask in mask_pyramid]

    L_blend_pyramid = []

    for level in range(NUM_LEVELS, -1, -1):
        # P_src = P_src_pyramid[level]
        L_src = L_src_pyramid[level]
        # P_tgt = P_tgt_pyramid[level]
        L_tgt = L_tgt_pyramid[level]

        # P_src_lap = signal.convolve2d(P_src, LAPLACIAN_FILTER, mode='same', boundary='symm')
        L_src_lap = signal.convolve2d(L_src, LAPLACIAN_FILTER, mode='same', boundary='symm')
        # P_tgt_lap = signal.convolve2d(P_tgt, LAPLACIAN_FILTER, mode='same', boundary='symm')
        # L_tgt_lap = signal.convolve2d(L_tgt, LAPLACIAN_FILTER, mode='same', boundary='symm')

        # mask_tgt, mask_border = create_target_mask(mask_pyramid[level], center, level, P_tgt.shape)
        ranges = get_src_and_tgt_ranges(center, level, L_src.shape, L_tgt.shape)
        mask = mask_pyramid[level][ranges["src_min_y"]:ranges["src_max_y"], ranges["src_min_x"]:ranges["src_max_x"]]
        mask_border = get_border_mask(mask)

        L_blend = np.zeros_like(L_tgt)

        #copy target image outside of the src 
        L_blend[:ranges["tgt_min_y"], :] = L_tgt[:ranges["tgt_min_y"], :]
        L_blend[ranges["tgt_max_y"]:, :] = L_tgt[ranges["tgt_max_y"]:, :]
        L_blend[:, :ranges["tgt_min_x"]] = L_tgt[:, :ranges["tgt_min_x"]]
        L_blend[:, ranges["tgt_max_x"]:] = L_tgt[:, ranges["tgt_max_x"]:]

        src_patch_in_tgt = L_tgt[ranges["tgt_min_y"]:ranges["tgt_max_y"], ranges["tgt_min_x"]:ranges["tgt_max_x"]]
        src_patch_in_im_blend = L_blend[ranges["tgt_min_y"]:ranges["tgt_max_y"], ranges["tgt_min_x"]:ranges["tgt_max_x"]]
        h_tgt, w_tgt = src_patch_in_tgt.shape

        im_vec = src_patch_in_im_blend.flatten()
        tgt_vec = src_patch_in_tgt.flatten()
        src_lap_vec = L_src_lap[ranges["src_min_y"]:ranges["src_max_y"], ranges["src_min_x"]:ranges["src_max_x"]].flatten()
        border_mask_vec = mask_border.flatten()
        mask_vec = mask.flatten()

        # building b
        b = np.zeros_like(im_vec)
        inside_vec = np.logical_and(mask_vec == 1, border_mask_vec == 0)
        b[inside_vec == False] = tgt_vec[inside_vec == False]
        b[inside_vec == True] = src_lap_vec[inside_vec == True]

        # building A
        #   diagonal:
        # inside_vec[:] = True
        A = create_laplacian(h_tgt, w_tgt, inside_vec)
        im_vec = spsolve(A, b)
        L_blend[ranges["tgt_min_y"]:ranges["tgt_max_y"], ranges["tgt_min_x"]:ranges["tgt_max_x"]] = im_vec.reshape((h_tgt, w_tgt))

        plt.imshow(L_blend, cmap='gray')
        plt.savefig("im_blend_{}.png".format(level))
        plt.close()

        L_blend_pyramid.append(cv2.resize(L_blend, (im_tgt.shape[1], im_tgt.shape[0])))


    im_blend = sum(L_blend_pyramid)
    return im_blend


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana2.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana2.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='mask file path')
    return parser.parse_args()

if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    plt.imshow(im_clone, cmap='gray')
    plt.savefig("im_blend.png")
    plt.close()

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
