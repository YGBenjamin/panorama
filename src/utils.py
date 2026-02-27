import numpy as np
import matplotlib.pyplot as plt

def rgb2gray(img):
    """Convertit une image RGB en niveaux de gris pour Harris et les descripteurs."""
    if img.ndim == 3:
        return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    return img

def warping(img, x_arr, y_arr):
    h, w = img.shape[:2]
    x0, y0 = np.clip(np.floor(x_arr).astype(int), 0, w-1), np.clip(np.floor(y_arr).astype(int), 0, h-1)
    x1, y1 = np.clip(x0+1, 0, w-1), np.clip(y0+1, 0, h-1)
    dx, dy = x_arr-x0, y_arr-y0
    
    # Adaptation pour le RGB (broadcasting des canaux)
    if img.ndim == 3:
        dx = dx[..., None]
        dy = dy[..., None]
        
    pixels = (1-dx)*(1-dy)*img[y0,x0] + dx*(1-dy)*img[y0, x1] + (1-dx)*dy*img[y1, x0] + dx*dy*img[y1, x1]
    return pixels

def homographie(img, H, h=None, w=None):
    if h == None or w == None : h, w = img.shape[:2]
    y_f, x_f = np.indices((h, w))
    n = h*w 
    P_dest = np.ones((3, n))
    P_dest[:-1, :] = np.array([x_f.flatten(), y_f.flatten()])

    P_src = np.linalg.inv(H) @ P_dest
    x_src = P_src[0] / P_src[2]
    y_src = P_src[1] / P_src[2]

    pixels_final = warping(img, x_src, y_src)
    
    # Adaptation pour RGB ou Gris
    if img.ndim == 3:
        img_final = pixels_final.reshape((h, w, img.shape[2]))
    else:
        img_final = pixels_final.reshape((h, w))
        
    return img_final

def convolution(image, C):
    C_n, C_m = C.shape[0]-1, C.shape[1]-1
    img_b = np.zeros((image.shape[0]+C_n, image.shape[1]+C_m)).astype(np.float32)
    decalage = C_n//2
    img_b[decalage:-decalage, decalage:-decalage] = image

    res = np.zeros_like(image).astype(np.float32)
    n, m = res.shape[0], res.shape[1]
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            res += C[i, j]*img_b[i:n+i, j:m+j]
    return res

def harris_detection(img, k=0.04, seuil=0.01):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = Kx.T
    
    Gx = convolution(img, Kx)
    Gy = convolution(img, Ky)

    Ixx = Gx*Gx
    Iyy = Gy*Gy
    Ixy = Gx*Gy

    blur = np.ones((3,3))/9.0
    Sxx = convolution(Ixx, blur)
    Syy = convolution(Iyy, blur)
    Sxy = convolution(Ixy, blur)

    det_m = Sxx*Syy - Sxy**2
    trace_m = Sxx + Syy
    R = det_m - k*(trace_m**2)

    seuil = seuil*R.max()
    corners = np.where(R > seuil)
    
    return corners

def descripteurs_vec(img, corners, patch_size=9):
    r_patch = patch_size//2
    h, w = img.shape[:2]
    y, x = corners
    mask = (y >= r_patch) & (y < h - r_patch) & (x >= r_patch) & (x < w - r_patch)
    y_v = y[mask]
    x_v = x[mask]
    n = len(y_v) 
    liste_descripteurs = np.ones((n, patch_size**2))
    compteur = 0
    for i in range(patch_size):
        for j in range(patch_size):
            pixel_of_all_patches = img[y_v-r_patch+i, x_v-r_patch+j] 
            liste_descripteurs[:, compteur] = pixel_of_all_patches
            compteur += 1
    mean_desc = np.mean(liste_descripteurs, axis=1, keepdims=True) 
    std_desc = np.std(liste_descripteurs, axis=1, keepdims=True)
    liste_descripteurs = (liste_descripteurs-mean_desc)/(std_desc+1e-8)
            
    return liste_descripteurs, np.column_stack((x_v, y_v))

def cdist(A, B):
    a_squared = np.sum(A**2, axis=1).reshape(-1, 1) 
    b_squared = np.sum(B**2, axis=1).reshape(1, -1) 
    dist = a_squared + b_squared - 2*(A@B.transpose()) 
    return np.sqrt(np.maximum(dist, 0)) 

def match_points(desc1, desc2, seuil_lowe, seuil_abs):
    distances = cdist(desc1, desc2)
    best_patches_idx = np.argpartition(distances, kth=2, axis=1)[:, :2] 
    lignes = np.arange(len(distances))
    d1 = distances[lignes, best_patches_idx[:, 0]]
    d2 = distances[lignes, best_patches_idx[:, 1]]

    lowe_ratios = d1/(d2+1e-8)
    mask = (lowe_ratios<=seuil_lowe) & (d1<=seuil_abs)
    indices_a = lignes[mask]
    indices_b = best_patches_idx[mask, 0]

    return np.column_stack((indices_a, indices_b))

def calculer_homographie(pts_src, pts_dst):
    A = [] 
    for i in range(len(pts_src)): 
        x, y = pts_src[i]
        xp, yp = pts_dst[i]
        A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])

    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    h = Vh[-1, :] 

    H = h.reshape((3, 3))
    return H / H[2, 2] 

def ransac(matching_points, corners1, corners2, n_iter=1000, seuil=3.0):
    if len(matching_points) < 4:
        return None, 0, []
    
    best_h = None
    max_inliers = 0
    best_inlier_indices = []

    pts_src = corners1[matching_points[:, 0]]
    pts_dst = corners2[matching_points[:, 1]] 

    for n in range(n_iter):
        random_idx = np.random.choice(len(pts_src), 4, replace=False)
        r_pts_src = pts_src[random_idx]
        r_pts_dst = pts_dst[random_idx]
        H = calculer_homographie(r_pts_src, r_pts_dst)

        non_projected_pts_homogenes = np.column_stack((pts_src, np.ones(len(pts_src))))
        projected_pts_homogenes = H@non_projected_pts_homogenes.T
        projected_pts = (projected_pts_homogenes[:2, :] / (projected_pts_homogenes[-1, :] + 1e-10)).T 

        dist = np.linalg.norm(projected_pts-pts_dst, axis=1) 
        mask = (dist < seuil)
        inliers = projected_pts[mask]
        nb_inliers = len(inliers)
        if nb_inliers > max_inliers:
            max_inliers = nb_inliers
            best_h = H
            best_inlier_indices = np.where(mask)[0] 
    
    return best_h, max_inliers, best_inlier_indices

def ransac_pipeline(img1, img2, seuil_harris=0.1, patch_size=9, lowe=0.7, seuil_abs=40, n_iter_ransac=1000, seuil_ransac=10, plot=False):
    # On travaille en niveaux de gris pour la détection
    img1_gray = rgb2gray(img1)
    img2_gray = rgb2gray(img2)
    
    corners_img1 = harris_detection(img1_gray, seuil=seuil_harris)
    corners_img2 = harris_detection(img2_gray, seuil=seuil_harris)
    
    liste_descripteurs_img1, valid_idx1 = descripteurs_vec(img1_gray, corners_img1, patch_size)
    liste_descripteurs_img2, valid_idx2 = descripteurs_vec(img2_gray, corners_img2, patch_size)

    matching_points = match_points(liste_descripteurs_img1, liste_descripteurs_img2, seuil_lowe=lowe, seuil_abs=seuil_abs)

    offset = img1.shape[1] 
    best_h, max_inliers, best_inlier_indices = ransac(matching_points, valid_idx1, valid_idx2, n_iter=n_iter_ransac, seuil=seuil_ransac)
    
    figs = [] # Pour stocker les figures pour Streamlit
    if plot and len(matching_points) > 0: 
        image_combinee = np.hstack((img1, img2))
        
        # Plot 1 : Sans RANSAC
        fig1 = plt.figure(figsize=(20, 10))
        # Si image float, on s'assure qu'elle s'affiche bien
        plt.imshow(image_combinee.astype(np.uint8) if image_combinee.max() > 1 else image_combinee, cmap='gray')
        for idx_A, idx_B in matching_points:
            xA, yA = valid_idx1[idx_A]
            xB, yB = valid_idx2[idx_B]
            plt.plot([xA, xB + offset], [yA, yB], alpha=0.5, linewidth=1)
            plt.scatter([xA, xB + offset], [yA, yB], s=10)
        plt.title(f"Nombre de matches trouvés : {len(matching_points)} (sans RANSAC)")
        figs.append(fig1)

        # Plot 2 : Avec RANSAC
        if best_h is not None:
            fig2 = plt.figure(figsize=(20, 10))
            plt.imshow(image_combinee.astype(np.uint8) if image_combinee.max() > 1 else image_combinee, cmap='gray')
            for idx_A, idx_B in matching_points[best_inlier_indices]:
                xA, yA = valid_idx1[idx_A]
                xB, yB = valid_idx2[idx_B]
                plt.plot([xA, xB + offset], [yA, yB], alpha=0.5, linewidth=1)
                plt.scatter([xA, xB + offset], [yA, yB], s=10)
            plt.title(f"Nombre de inliers trouvés : {max_inliers} (avec RANSAC)")
            figs.append(fig2)

    return (np.linalg.inv(best_h) if best_h is not None else None), figs

def create_panorama(img_L, img_R, H):
    h1, w1 = img_L.shape[:2]
    h2, w2 = img_R.shape[:2]

    corners = np.array([[0, 0], [w2-1, 0], [0, h2-1], [w2-1, h2-1]]).T
    corners_h = H @ np.vstack((corners, np.ones((1, 4))))
    x_h = corners_h[0] / corners_h[2]
    y_h = corners_h[1] / corners_h[2]

    x_min, x_max = min(0, np.min(x_h)), max(w1, np.max(x_h))
    y_min, y_max = min(0, np.min(y_h)), max(h1, np.max(y_h))

    largeur = int(np.ceil(x_max - x_min))
    hauteur = int(np.ceil(y_max - y_min))

    offset_x = int(round(-x_min))
    offset_y = int(round(-y_min))

    T = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]])
    H_final = T @ H

    panorama = homographie(img_R, H_final, h=hauteur, w=largeur)
    
    # Slicing robuste pour éviter l'erreur de broadcast !
    y_start, y_end = offset_y, min(offset_y + h1, hauteur)
    x_start, x_end = offset_x, min(offset_x + w1, largeur)
    
    # On superpose l'image de gauche
    panorama[y_start:y_end, x_start:x_end] = img_L[:(y_end-y_start), :(x_end-x_start)]

    return panorama
    
def panorama_pipeline(img1, img2, seuil_harris=0.1, patch_size=9, lowe=0.7, seuil_abs=40, n_iter_ransac=1000, seuil_ransac=10, plot=False):
    best_H, figs = ransac_pipeline(img1, img2, seuil_harris, patch_size, lowe, seuil_abs, n_iter_ransac, seuil_ransac, plot)
    
    if best_H is None:
        return None, figs # Pas assez de points pour faire le panorama
        
    img_final = create_panorama(img1, img2, best_H)
    return img_final, figs