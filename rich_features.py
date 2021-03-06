import cv2
import numpy as np

class Point3D(object):
    def __init__(self, coords, origin):
        self.coords = coords
        self.origin = origin

class Point2D(object):
    def __init__(self, view, coords):
        self.view = view
        self.coords = coords

def find_keypoints_descriptors(img):
    '''Detects keypoints and computes their descriptors.'''
    # initiate detector with a specified Hessian threshold (default: 100)
    detector = cv2.xfeatures2d.SURF_create(400)
    #detector.hessianThreshold = 100.0
    #detector.extended = False

    # find the keypoints and descriptors
    kp, des = detector.detectAndCompute(img, None)

    return kp, des

def match_keypoints(kp1, des1, kp2, des2):
    '''Matches the descriptors in one image with those in the second image using
    the Fast Library for Approximate Nearest Neighbours (FLANN) matcher.'''
    MIN_MATCH_COUNT = 10

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)

    # store all the good matches as per Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
    else:
        print "Not enough matches were found - %d/%d" % (len(good_matches), MIN_MATCH_COUNT)

    # src_pts and dst_pts are Nx1x2 arrays that contain the x and y pixel coordinates
    return src_pts, dst_pts

def filter_keypoints(mask, src_pts, dst_pts):
    '''Filter the keypoints using the mask of inliers generated by findFundamentalMat.'''
    # src_pts and dst_pts are Nx1x2 arrays that contain the x and y pixel coordinates
    src_pts = src_pts[mask.ravel()==1]
    dst_pts = dst_pts[mask.ravel()==1]

    return src_pts, dst_pts

def attach_indices(i, pts_3D, src_pts, dst_pts, pt_cloud_indexed=[]):
    '''Attach to each 3D point, indices into the original lists of keypoints and descriptors
    of the 2D points that contributed to this 3D point in the cloud.'''

    def find_point(new_pt, pt_cloud_indexed):
        for old_pt in pt_cloud_indexed:
            try:
                if np.array_equal(new_pt.origin[i], old_pt.origin[i]):
                    return True, old_pt
            except KeyError:
                continue
        return False, None

    new_pts = [ Point3D(pt, {i: src_pts[num], i+1: dst_pts[num]}) for num, pt in enumerate(pts_3D) ]

    if pt_cloud_indexed == []:
        pt_cloud_indexed = new_pts
    else:
        for num, new_pt in enumerate(new_pts):
            found, old_pt = find_point(new_pt, pt_cloud_indexed)
            if found:
                old_pt.origin[i+1] = dst_pts[num]
            else:
                pt_cloud_indexed.append(new_pt)

    return pt_cloud_indexed

def scan_cloud(i, prev_dst, src_pts, pt_cloud_indexed):
    '''Check for matches between the new frame and the current point cloud.'''
    # prev_dst contains the x & y coords of the keypoints from the second image in the last iteration
    # src_pts contains the x & y coords of the keypoints from the first image in the current iteration
    # the second image in the last iteration is the first image in the current iteration
    # therefore, check for matches by comparing the x & y coords
    matched_pts_2D = []
    matched_pts_3D = []
    indices = []

    for idx, new_pt in enumerate(src_pts):
        for old_pt in prev_dst:
            if np.array_equal(new_pt, old_pt):
                # found a match: a keypoint that contributed to both the last and current point clouds
                matched_pts_2D.append(new_pt)
                indices.append(idx)

    for pt_2D in matched_pts_2D:
        # pt_cloud_indexed is a list of 3D points from the previous cloud with their 2D pixel origins
        for pt in pt_cloud_indexed:
            try:
                if np.array_equal(pt.origin[i], pt_2D):
                    matched_pts_3D.append( pt.coords )
                    break
            except KeyError:
                continue
        continue

    matched_pts_2D = np.array(matched_pts_2D, dtype='float32')
    matched_pts_3D = np.array(matched_pts_3D, dtype='float32')

    return matched_pts_2D, matched_pts_3D, indices
