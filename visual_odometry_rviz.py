import numpy as np
import cv2
import math
import rospy
from matplotlib import pyplot as plt
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from numpy.linalg import inv
poses = [np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]], dtype=float)]
lk_params = dict(winSize  = (21, 21), 
				#maxLevel = 3,
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
class Point3D(object):
    def __init__(self, coords, origin):
        self.coords = coords
        self.origin = origin
class Point2D(object):
    def __init__(self, view, coords):
        self.view = view
        self.coords = coords
def magnitude(vector):
    return math.sqrt(vector[0]**2 + vector[1]**2)

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



def match_points(img2, flow):
    '''Finds points in the second image that matches the first, based on the motion flow vectors.'''
    # min and max magnitudes of the motion flow vector to be included in the reconstruction
    MIN_MAG, MAX_MAG = 0.5, 100
    # create an empty HxW array to store the dst points
    h, w = img2.shape[0], img2.shape[1]

    src_pts = [ [[col, row]] for row in xrange(h) for col in xrange(w) if (0 < int(row + flow[row, col][0]) < h) and (0 < int(col + flow[row, col][1]) < w) and MIN_MAG < magnitude(flow[row, col]) < MAX_MAG ]
    dst_pts = [ [[int(col + flow[row, col][1]), int(row + flow[row, col][0])]] for row in xrange(h) for col in xrange(w) if (0 < int(row + flow[row, col][0]) < h) and (0 < int(col + flow[row, col][1]) < w) and MIN_MAG < magnitude(flow[row, col]) < MAX_MAG ]
    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)
    # src and dst pts are Nx1x2 arrays that contain the x and y coordinates of the matching points
    print('source points',src_pts)
    print('destination points',dst_pts)
    return src_pts, dst_pts

def triangulate_points(P1,P2,refined_pts1,refined_pts2):
        norm_pts1=refined_pts1[0].T
        norm_pts2=refined_pts2[0].T
        homog_3D = cv2.triangulatePoints(P1, P2,norm_pts1,norm_pts2)
        #print("number of zeros in homog_3D",np.where(homog_3D[3]== 0))         #TODO Remove all 3D points whose last element is zero
        homog_3D = homog_3D / homog_3D[3]
        pts_3D = np.array(homog_3D[:3]).T   
        return homog_3D,pts_3D
    


def compute_cam_pose(K_matrices, matched_pts_2D, matched_pts_3D, poses,distortion):
    '''Compute the camera pose from a set of 3D and 2D correspondences.'''
    #K1 = K_matrices[-2]
    retval,rvec, tvec,inliers = cv2.solvePnPRansac(matched_pts_3D, matched_pts_2D, K_matrices, distortion)
    print("Rvector",rvec)
    rmat = cv2.Rodrigues(rvec)[0]
    print("Rvector after trans",rmat)
    pose = np.hstack((rmat, tvec))
    print(np.shape(pose))
    poses.append(pose)
    return poses


def feature_matching_tracking(i,uframe1,uframe2,poses,K):
        gray1=cv2.cvtColor(uframe1,cv2.COLOR_RGB2GRAY)
        gray2=cv2.cvtColor(uframe2,cv2.COLOR_RGB2GRAY)
        kp1, des1 = find_keypoints_descriptors(gray1)
        kp2, des2 = find_keypoints_descriptors(gray2)
        points1,points2 = match_keypoints(kp1, des1, kp2, des2)
        #print("Point1",points1)
        #print("Point2",points2)
        E,mask=cv2.findEssentialMat(points1,points2,focal=1110.25,pp=(640,298),method=cv2.RANSAC,prob=0.99)           
        _,R,t,mask=cv2.recoverPose(E,points1,points2,focal=1110.25,pp=(640,298))
        P1=poses[-1]
        P2=[]
        P2=np.concatenate((R,t),axis=1)
        homog_3D,pts_3D=triangulate_points(P1,P2,points1,points2)
        pt_cloud_indexed = attach_indices(i, pts_3D, points1,points2)
        return points2,homog_3D, pts_3D,pt_cloud_indexed


def cameraposes_from_multiple_views(i,K_matrices,uframe1,uframe2,prev_dst,poses,pt_cloud_indexed,distortion,last):
	gray1=cv2.cvtColor(uframe1,cv2.COLOR_RGB2GRAY)
        gray2=cv2.cvtColor(uframe2,cv2.COLOR_RGB2GRAY)
        prev_kp, prev_des = find_keypoints_descriptors(gray1)
        new_kp, new_des = find_keypoints_descriptors(gray2)
        points1,points2 = match_keypoints(prev_kp, prev_des, new_kp, new_des)
        #print("Point1",points1)
        #print("Point2",points2)        
        E,mask=cv2.findEssentialMat(points1,points2,focal=1110.25,pp=(640,298),method=cv2.RANSAC,prob=0.99)
        matched_pts_2D, matched_pts_3D, indices = scan_cloud(i, prev_dst,points1, pt_cloud_indexed)
        print("matched pts 2d",matched_pts_2D)
        print("matched pts 3d",matched_pts_3D)
        poses = compute_cam_pose(K_matrices,matched_pts_2D,matched_pts_3D, poses,distortion)
        _,R,t,mask=cv2.recoverPose(E,points1,points2,focal=1110.25,pp=(640,298))
        P1=poses[-1]
        print("poses",P1)
        P2=[]
        P2=np.concatenate((R,t),axis=1)
	homog_3D, pts_3D = triangulate_points(P1, P2,points1,points2)
        pt_cloud_indexed = attach_indices(i, pts_3D, points1, points2,pt_cloud_indexed)
        return homog_3D, pts_3D,pt_cloud_indexed,points2, poses

def start():
        i=0
        rospy.init_node('Pose_node', anonymous=True)
        pub_vision = rospy.Publisher('/vision/pose', PoseStamped, queue_size=10)
        state = PoseStamped()
        state.header.seq=0
        state.header.stamp=rospy.Time.now()
        state.header.frame_id="map"
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
                prev_dst,homog_3D, pts_3D,pt_cloud_indexed=feature_matching_tracking(i,frame1,frame2,poses,mtx)                
                pt_cloud = np.array(pts_3D)
                print("Exiting first phase")
                           
        while(cap.isOpened()):
                state.header.stamp=rospy.Time.now()
                state.header.seq = state.header.seq+1                
                i=i+1
                print("VALUE OF I",i)     
                ret,current_frame=cap.read()
                temp.append(current_frame)
                if(i>=1):
                        print("VALUE OF I",i)
                        print("Entering second phase")
                        homog_3D, pts_3D,pt_cloud_indexed,prev_dst,poses=cameraposes_from_multiple_views(i,mtx,temp[i-1],temp[i],prev_dst,poses,pt_cloud_indexed,distortion,last=False)
                        print(np.shape(poses))
                        poses1=np.resize(poses,(4,4))
                        
                        state.pose.position.x = poses1[0][3]
                        state.pose.position.y = poses1[1][3]
                        state.pose.position.z = poses1[2][3]
                        sy = math.sqrt(poses1[0,0] * poses1[0,0] +  poses1[1,0] * poses1[1,0])
                        singular = sy < 1e-6
                        if  not singular :
                                x = math.atan2(poses1[2,1] , poses1[2,2])
                                y = math.atan2(-poses1[2,0], sy)
                                z = math.atan2(poses1[1,0], poses1[0,0])
                        else :
                                x = math.atan2(-poses1[1,2], poses1[1,1])
                                y = math.atan2(-poses1[2,0], sy)
                                z = 0
                        q = tf.transformations.quaternion_from_euler(x,y,z)
                        print("Q",q)    
                        state.pose.orientation.x = q[0]
                        state.pose.orientation.y = q[1]
                        state.pose.orientation.z = q[2]
                        state.pose.orientation.w = q[3]
                        pub_vision.publish(state)
                        pt_cloud = np.vstack((pt_cloud, pts_3D))
                        print("Exiting second phase")
           
                
             
		
if __name__ == '__main__':

	try:
		start()
	except rospy.ROSInterruptException:
		pass
        

cap.release()
out.release()
cv2.destroyAllWindows
