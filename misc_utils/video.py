# @AdamSpannbauer
import cv2
import imutils.feature.factories as kp_factory
import numpy as np


def match_keypoints(optical_flow, prev_kps):
    """Match optical flow keypoints
    :param optical_flow: output of cv2.calcOpticalFlowPyrLK
    :param prev_kps: keypoints that were passed to cv2.calcOpticalFlowPyrLK to create optical_flow
    :return: tuple of (cur_matched_kp, prev_matched_kp)
    """
    cur_kps, status, err = optical_flow

    # storage for keypoints with status 1
    prev_matched_kp = []
    cur_matched_kp = []
    for i, matched in enumerate(status):
        # store coords of keypoints that appear in both
        if matched:
            prev_matched_kp.append(prev_kps[i])
            cur_matched_kp.append(cur_kps[i])

    return cur_matched_kp, prev_matched_kp


def estimate_partial_transform(matched_keypoints):
    """Wrapper of cv2.estimateRigidTransform for convenience in vidstab process
    :param matched_keypoints: output of match_keypoints util function; tuple of (cur_matched_kp, prev_matched_kp)
    :return: transform as list of [dx, dy, da]
    """
    cur_matched_kp, prev_matched_kp = matched_keypoints

    transform = cv2.estimateRigidTransform(np.array(prev_matched_kp),
                                           np.array(cur_matched_kp),
                                           False)
    if transform is not None:
        # translation x
        dx = transform[0, 2]
        # translation y
        dy = transform[1, 2]
        # rotation
        da = np.arctan2(transform[1, 0], transform[0, 0])
    else:
        dx = dy = da = 0

    return [dx, dy, da]


def video_trajectory(video):
    kp_detector = kp_factory.FeatureDetector_create('GFTT',
                                                    maxCorners=200,
                                                    qualityLevel=0.01,
                                                    minDistance=30.0,
                                                    blockSize=3)
    prev_frame = video[0]
    trajectory = [np.zeros((3,))]
    for frame in video[1:]:
        prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_kps = kp_detector.detect(prev_frame_gray)
        prev_kps = np.array([kp.pt for kp in prev_kps], dtype='float32').reshape(-1, 1, 2)
        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        optical_flow = cv2.calcOpticalFlowPyrLK(prev_frame_gray,
                                                current_frame_gray,
                                                prev_kps, None)

        matched_keypoints = match_keypoints(optical_flow, prev_kps)
        transform_i = estimate_partial_transform(matched_keypoints)
        trajectory.append(trajectory[-1] + transform_i)
    return trajectory


def trajectory_loss(trajectory):
    loss = 0
    for t1, t2 in zip(trajectory, trajectory[1:]):
        loss += (t2 - t1).abs().sum()
    return loss
