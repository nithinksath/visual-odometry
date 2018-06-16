import sys
import numpy as np

from processing import *
from rich_features import *


def gen_pt_cloud(i,image1, image2, poses,K_matrices):
    '''Generates a point cloud for a pair of images.'''
    print "    Loading images..."

    img1_gray, img2_gray = gray_images(image1, image2)



    # use rich feature matching to compute point correspondences

    print "    Detecting keypoints...\n    Computing descriptors..."
    kp1, des1 = find_keypoints_descriptors(img1_gray)
    kp2, des2 = find_keypoints_descriptors(img2_gray)
    print "    Matching keypoints..."
    src_pts, dst_pts = match_keypoints(kp1, des1, kp2, des2)
    norm_pts1, norm_pts2 = normalize_points(K_matrices, src_pts, dst_pts)

    print "    Finding the essential and projection matrices..."
    E, mask = find_essential_matrix(K_matrices, norm_pts1, norm_pts2)
    P1, P2 = find_projection_matrices(E, poses)
    src_pts, dst_pts = filter_keypoints(mask, src_pts, dst_pts)
    norm_pts1, norm_pts2 = apply_mask(mask, norm_pts1, norm_pts2)
    refined_pts1, refined_pts2 = refine_points(norm_pts1, norm_pts2, E)

    print "    Triangulating 3D points..."
    homog_3D, pts_3D, infront = triangulate_points(P1, P2, refined_pts1, refined_pts2)
    norm_pts1, norm_pts2 = apply_infront_filter(infront, norm_pts1, norm_pts2)


    print "    Initializing feature tracks..."
    pt_cloud_indexed = attach_indices(i, pts_3D, src_pts, dst_pts)

    return dst_pts, homog_3D, pts_3D,pt_cloud_indexed




def find_new_pts_feat(i,K_matrices,image1, image2, prev_dst, poses, pt_cloud_indexed):
    print "    Loading images..."

    img1_gray, img2_gray = gray_images(image1, image2)


    print "    Detecting keypoints...\n    Computing descriptors..."
    prev_kp, prev_des = find_keypoints_descriptors(img1_gray)
    new_kp, new_des = find_keypoints_descriptors(img2_gray)
    print "    Matching keypoints..."
    src_pts, dst_pts = match_keypoints(prev_kp, prev_des, new_kp, new_des)
    norm_pts1, norm_pts2 = normalize_points(K_matrices, src_pts, dst_pts)

    E, mask = find_essential_matrix(K_matrices, norm_pts1, norm_pts2)
    src_pts, dst_pts = filter_keypoints(mask, src_pts, dst_pts)
    norm_pts1, norm_pts2 = apply_mask(mask, norm_pts1, norm_pts2)
    refined_pts1, refined_pts2 = refine_points(norm_pts1, norm_pts2, E)

    print "    Scanning cloud..."
    matched_pts_2D, matched_pts_3D, indices = scan_cloud(i, prev_dst, src_pts, pt_cloud_indexed)
    print "    Computing camera pose..."
    poses = compute_cam_pose(K_matrices, matched_pts_2D, matched_pts_3D, poses)
    P1, P2 = find_projection_matrices(E, poses)

    print "    Triangulating 3D points..."
    homog_3D, pts_3D, infront = triangulate_points(P1, P2, refined_pts1, refined_pts2)
    norm_pts1, norm_pts2 = apply_infront_filter(infront, norm_pts1, norm_pts2)


    print "    Assembling feature tracks..."
    pt_cloud_indexed = attach_indices(i, pts_3D, src_pts, dst_pts, pt_cloud_indexed)



    return  dst_pts, poses, homog_3D, pts_3D, pt_cloud_indexed





def start():
    '''Loop through each pair of images, find point correspondences and generate 3D point cloud.
    For each new frame, find additional points and add them to the overall point cloud.'''


    i=0
    temp=[]
    uframe=[]
    cap=cv2.VideoCapture(0)
    poses = [np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]], dtype=float)]
    mtx=np.loadtxt('/home/sakthi/Nithin/real_calibration/proper_calibration/images/rgb/cameramatrix.txt')
    distortion=np.loadtxt('/home/sakthi/Nithin/real_calibration/proper_calibration/images/rgb/distortioncoefficient.txt')
    ret,frame1=cap.read()
    ret,frame2=cap.read()
    prev_frame=frame2
    temp.append(prev_frame)
    if(i==0):
        print("Entering first phase")
        prev_dst,homog_3D, pts_3D,pt_cloud_indexed=gen_pt_cloud(i,frame1,frame2,poses,mtx)
        pt_cloud = np.array(pts_3D)
        print("Exiting first phase")

    while(cap.isOpened()):
        i=i+1
        ret,current_frame=cap.read()
        temp.append(current_frame)
        if(i>=1):
            prev_dst,poses,homog_3D, pts_3D,pt_cloud_indexed= find_new_pts_feat(i,mtx,temp[i-1],temp[i],prev_dst,poses,pt_cloud_indexed)
            pt_cloud = np.vstack((pt_cloud, pts_3D))




    # homog_pt_cloud = np.vstack((pt_cloud.T, np.ones(pt_cloud.shape[0])))
    # draw.draw_matches(src_pts, dst_pts, img1_gray, img2_gray)
    # draw.draw_epilines(src_pts, dst_pts, img1_gray, img2_gray, F, mask)
    # draw.draw_projected_points(homog_pt_cloud, P)
    # display_vtk.vtk_show_points(pt_cloud, list(colours))



if __name__ == "__main__":
    start()
