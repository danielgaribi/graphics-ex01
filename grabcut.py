import numpy as np
import cv2
import argparse
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import igraph as ig

np.seterr(divide = 'ignore')

GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel

epsilon = 0.00001

n_links = None
n_links_weights = None
K = None
prev_energy = None

# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    # Convert to absolute cordinates
    w -= x
    h -= y
    img_float = img.astype(np.float64)

    #Initalize the inner square to Foreground
    mask[y:y+h, x:x+w] = GC_PR_FGD
    mask[rect[1]+rect[3]//2, rect[0]+rect[2]//2] = GC_FGD

    bgGMM, fgGMM = initalize_GMMs(img_float, mask)

    num_iters = 1000
    for i in range(num_iters):
        #Update GMM
        bgGMM, fgGMM = update_GMMs(img_float, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img_float, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy):
            break

    mask = finalize_mask(mask)

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM

def finalize_mask(mask):
    mask[mask == GC_PR_BGD] = GC_BGD
    mask[mask == GC_PR_FGD] = GC_FGD
    return mask

def initalize_GMMs(img, mask, n_components = 5):
    background_pixels = img[np.logical_or(mask == GC_PR_BGD, mask == GC_BGD)].reshape(-1, 3)
    foreground_pixels = img[np.logical_or(mask == GC_PR_FGD, mask == GC_FGD)].reshape(-1, 3)

    kmeans_background = KMeans(n_clusters=n_components, n_init=10).fit(background_pixels)
    kmeans_foreground = KMeans(n_clusters=n_components, n_init=10).fit(foreground_pixels)

    bgGMM = GaussianMixture(n_components=n_components)
    bgGMM.weights_ = np.ones(n_components) / n_components
    bgGMM.means_ = kmeans_background.cluster_centers_
    bgGMM.covariances_ = np.array([np.eye(3) * (1 + epsilon) ] * n_components)
    background_pixels_prediction = kmeans_background.predict(background_pixels)
    for comp_index in range(n_components):
        comp_pixels = background_pixels[background_pixels_prediction==comp_index]
        bgGMM.covariances_[comp_index] = np.cov(comp_pixels.T)
    bgGMM.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(bgGMM.covariances_)).transpose((0, 2, 1))
    
    fgGMM = GaussianMixture(n_components=n_components)
    fgGMM.weights_ = np.ones(n_components) / n_components
    fgGMM.means_ = kmeans_foreground.cluster_centers_
    fgGMM.covariances_ = np.array([np.eye(3) * (1 + epsilon)] * n_components)
    foreground_pixels_prediction = kmeans_foreground.predict(foreground_pixels)
    for comp_index in range(n_components):
        comp_pixels = foreground_pixels[foreground_pixels_prediction==comp_index]
        fgGMM.covariances_[comp_index] = np.cov(comp_pixels.T)
    fgGMM.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(fgGMM.covariances_)).transpose((0, 2, 1))

    return bgGMM, fgGMM

# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    # Get foreground and background pixels from mask
    background_pixels = img[np.logical_or(mask == GC_PR_BGD, mask == GC_BGD)].reshape(-1, 3)
    foreground_pixels = img[np.logical_or(mask == GC_PR_FGD, mask == GC_FGD)].reshape(-1, 3)

    n_components = fgGMM.weights_.shape[0]

    # Update background GMMs
    background_pixels_prediction = bgGMM.predict(background_pixels)
    for comp_index in range(n_components):
        comp_pixels = background_pixels[background_pixels_prediction==comp_index]
        # In the case of empty components, do not update
        if comp_pixels.shape[0] == 0:
            continue
        bgGMM.means_[comp_index] = np.mean(comp_pixels, axis=0)
        bgGMM.covariances_[comp_index] = np.cov(comp_pixels.T) + np.eye(3) * (epsilon)
        bgGMM.weights_[comp_index] = comp_pixels.shape[0] / background_pixels.shape[0]

    bgGMM.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(bgGMM.covariances_)).transpose((0, 2, 1))

    # Update foreground GMMs
    foreground_pixels_prediction = fgGMM.predict(foreground_pixels)
    for comp_index in range(n_components):
        comp_pixels = foreground_pixels[foreground_pixels_prediction==comp_index]
        if comp_pixels.shape[0] == 0:
            continue
        fgGMM.means_[comp_index] = np.mean(comp_pixels, axis=0)
        fgGMM.covariances_[comp_index] = np.cov(comp_pixels.T) + np.eye(3) * (epsilon)
        fgGMM.weights_[comp_index] = comp_pixels.shape[0] / foreground_pixels.shape[0]

    fgGMM.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(fgGMM.covariances_)).transpose((0, 2, 1))

    return bgGMM, fgGMM

def calc_beta(diff_right, diff_bottom, diff_bottom_right, diff_top_right):
    rows, cols, _ = diff_right.shape

    sum_of_all_diffs = np.sum(np.square(diff_right)) + \
                       np.sum(np.square(diff_bottom)) + \
                       np.sum(np.square(diff_bottom_right)) + \
                       np.sum(np.square(diff_top_right))
    
    total_nof_elements = 4 * cols * rows - 3 * rows - 3 * cols + 2
    exp = sum_of_all_diffs / total_nof_elements
    return 1 / (2 * exp)

def calculate_n_links(img):
    rows, cols = img.shape[:2]
    indices = np.arange(rows * cols).reshape(rows, cols)

    # Initialize matrices
    diff_right = np.zeros_like(img)
    diff_bottom = np.zeros_like(img)
    diff_bottom_right = np.zeros_like(img)
    diff_top_right = np.zeros_like(img)

    # Calculate differences in pixel values
    diff_right[:,:-1]           = img[:, :-1]   - img[:, 1:]
    diff_bottom[:-1, :]         = img[:-1, :]   - img[1:, :]
    diff_bottom_right[:-1, :-1] = img[:-1, :-1] - img[1:, 1:]
    diff_top_right[1:, :-1]     = img[1:, :-1]  - img[:-1, 1:]

    # Calculate norms of pixel value differences
    norm_right = np.linalg.norm(diff_right, axis=2)
    norm_bottom = np.linalg.norm(diff_bottom, axis=2)
    norm_bottom_right = np.linalg.norm(diff_bottom_right,axis=2)
    norm_top_right = np.linalg.norm(diff_top_right, axis=2)
    
    beta = calc_beta(diff_right, diff_bottom, diff_bottom_right, diff_top_right)

    # Calculate weights
    weight_right = (50 * np.exp(-beta * np.power(norm_right, 2))) / 1
    weight_bottom = (50 * np.exp(-beta * np.power(norm_bottom, 2))) / 1
    weight_bottom_right = (50 * np.exp(-beta * np.power(norm_bottom_right, 2))) / np.sqrt(2)
    weight_top_right = (50 * np.exp(-beta * np.power(norm_top_right, 2))) / np.sqrt(2)

    # Update sum of weights per pixel
    sum_weights_per_pix = np.zeros((rows, cols))
    sum_weights_per_pix += 2 * weight_right         # multipy by 2 because we are adding the weight from the right and left
    sum_weights_per_pix += 2 * weight_bottom        # multipy by 2 because we are adding the weight from the top and bottom
    sum_weights_per_pix += 2 * weight_bottom_right  # multipy by 2 because we are adding the weight from the top-left and bottom-right
    sum_weights_per_pix += 2 * weight_top_right     # multipy by 2 because we are adding the weight from the top-right and bottom-left

    # Extract n_links and n_links_weights
    n_links = []
    n_links_weights = []
    for y in range(rows):
        for x in range(cols):
            if x < cols - 1:
                n_links.append((indices[y, x], indices[y, x+1]))
                n_links_weights.append(weight_right[y, x])

            if y < rows - 1:
                n_links.append((indices[y, x], indices[y+1, x]))
                n_links_weights.append(weight_bottom[y, x])

            if x < cols - 1 and y < rows - 1:
                n_links.append((indices[y, x], indices[y+1, x+1]))
                n_links_weights.append(weight_bottom_right[y, x])

            if x < cols - 1 and y > 0:
                n_links.append((indices[y, x], indices[y-1, x+1]))
                n_links_weights.append(weight_top_right[y, x])

    max_weight = np.max(sum_weights_per_pix)

    return n_links, n_links_weights, max_weight

def calc_D_for_image(pixels, gmm):
    log_prob = np.zeros((pixels.shape[0], 1))
    for i in range(gmm.n_components):
        coef = gmm.weights_[i]
        mean = gmm.means_[i]
        covar = gmm.covariances_[i]
        diff = pixels - mean
        diff = diff.reshape(-1, 3, 1)
        covarDet = np.linalg.det(covar)

        mul_cov_diff = np.einsum("i j, b j k -> b i k", np.linalg.inv(covar) , diff)
        exponent = -0.5 * np.einsum("b k j, b j i -> b k i", diff.transpose(0, 2, 1), mul_cov_diff)
        exponent = exponent.reshape(-1, 1)
        log_prob += ( coef / np.sqrt(covarDet) ) * np.exp(exponent)
    
    log_prob = -1 * np.log(log_prob)
    return log_prob

def calculate_t_links(img, mask, bgGMM, fgGMM, bg_node, fg_node, K):
    rows, cols = img.shape[:2]
    indices = np.arange(rows * cols).reshape(rows, cols)

    bg_weights = calc_D_for_image(img.reshape(-1, 3), fgGMM).reshape(img.shape[:-1])
    fg_weights = calc_D_for_image(img.reshape(-1, 3), bgGMM).reshape(img.shape[:-1])

    bg_weights[mask == GC_BGD] = K
    bg_weights[mask == GC_FGD] = 0

    fg_weights[mask == GC_BGD] = 0
    fg_weights[mask == GC_FGD] = K

    t_links = []
    t_links_weights = []
    for y in range(rows):
        for x in range(cols):
            i = indices[y, x]
            t_links.append((i, bg_node))
            t_links_weights.append(bg_weights[(y,x)])
            t_links.append((i, fg_node))
            t_links_weights.append(fg_weights[(y,x)])

    return t_links, t_links_weights

def build_graph(img, mask, bgGMM, fgGMM, bg_node, fg_node):
    global n_links, n_links_weights, K
    rows, cols = img.shape[:2]
    graph = ig.Graph()
    graph.add_vertices(rows * cols + 2)  # 2 extra vertices for source and sink

    if n_links is None or n_links_weights is None or K is None:
        n_links, n_links_weights, K = calculate_n_links(img)

    
    t_links, t_links_weights = calculate_t_links(img, mask, bgGMM, fgGMM, bg_node, fg_node, K)

    graph.add_edges(n_links + t_links, attributes={'weight': n_links_weights + t_links_weights})
    return graph

def calculate_mincut(img, mask, bgGMM, fgGMM):
    # Build graph
    bg_node = img.shape[0] * img.shape[1]
    fg_node = bg_node + 1
    graph = build_graph(img, mask, bgGMM, fgGMM, bg_node, fg_node)
    
    # Find the minimum cut
    cut = ig.Graph.st_mincut(graph, bg_node, fg_node, capacity='weight')

    # Get the min_cut
    min_cut = cut.partition
    for i in range(len(min_cut)):
        min_cut[i] = [(node//img.shape[1], node % img.shape[1]) for node in min_cut[i] if node<bg_node]

    # Get the energy term corresponding to the cut
    energy = cut.value
    return min_cut, energy

def update_mask(mincut_sets, mask):
    # width, height = np.shape(mask)
    background_pixels = mincut_sets[0]
    foreground_pixels = mincut_sets[1]

    for bg_pixel_ind in background_pixels:
        if mask[bg_pixel_ind] != GC_BGD and mask[bg_pixel_ind] != GC_FGD:
            mask[bg_pixel_ind] = GC_PR_BGD

    for fg_pixel_ind in foreground_pixels:
        if mask[fg_pixel_ind] != GC_BGD and mask[fg_pixel_ind] != GC_FGD:
            mask[fg_pixel_ind] = GC_PR_FGD
    
    return mask

def check_convergence(energy):
    global prev_energy 
    threshold = 0.0005

    # If prev_energy is None, set it to the current energy and return False
    if prev_energy is None:
        prev_energy = energy
        return False

    delta = abs(energy - prev_energy)
    normlized_delta = delta / (energy + epsilon)
    
    if (energy > prev_energy or normlized_delta < threshold):
        return True
    
    prev_energy = energy
    return False

def cal_metric(predicted_mask, gt_mask):
    # Convert the masks to boolean arrays for element-wise operations
    predicted_mask = predicted_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    # Calculate accuracy
    accuracy = (predicted_mask == gt_mask).mean() * 100

    # Calculate intersection, union, and Jaccard similarity
    intersection = (predicted_mask & gt_mask).sum()
    union = (predicted_mask | gt_mask).sum()
    jaccard_similarity = (intersection / union) * 100

    return accuracy, jaccard_similarity

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana1', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()

if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()

    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int,args.rect.split(',')))

    img = cv2.imread(input_path)

    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
