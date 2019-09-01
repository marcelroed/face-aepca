import cv2
import numpy as np
import face_recognition
from tqdm import tqdm
from typing import *

from skimage import transform

IDEAL_POS = np.array([32, 21])
IDEAL_SIZE = 24
IMAGE_SHAPE = (64, 64)

data_dir = '../../data/famousbirthdays/'


def clean_data():
    images = np.load(f'{data_dir}images.npy')
    names = np.load(f'{data_dir}names.npy')

    # Find faces with features in image
    face_landmarks = [face_recognition.face_landmarks(image) for image in tqdm(images, desc='Finding face landmarks')]
    exactly_one_face = np.array([len(lms) == 1 for lms in face_landmarks], dtype=bool)

    # Check if images are grayscale or colored
    is_colored = np.array([colored(image) for image in tqdm(images, desc='Checking if images are grayscale')])

    # Mask only valid images
    mask = is_colored & exactly_one_face
    face_landmarks = np.array([lms[0] if len(lms) else np.NaN for lms in face_landmarks])
    images = images[mask]; names = names[mask]
    face_landmarks = np.array(face_landmarks[mask])

    # Find the average face position
    avg_face = find_average_features(face_landmarks)

    # Translate, scale, rotate and warp faces to match average face
    warped, mask = warp_faces_to_features(images, face_landmarks, avg_face)
    warped = warped[mask]; names = names[mask]

    np.save(f'{data_dir}processed_images.npy', warped)
    np.save(f'{data_dir}processed_names.npy', names)


def get_mean_positions(face_features: np.array):
    mean_positions = {}
    for key, val in face_features.items():
        mean_positions[key] = np.mean(np.array(val), axis=0)
    return mean_positions


def find_average_features(all_features: np.array):
    keys = all_features[0].keys()
    feat_mean = {}
    for key in keys:
        feat_mean[key] = np.mean(np.array([np.mean(np.array(feat[key]), axis=0) for feat in all_features]), axis=0)
    '''
    print(feat_mean)
    img = np.zeros(shape=(190, 190, 3), dtype=np.uint8)
    for feat in feat_mean.values():
        img = cv2.circle(img, tuple(feat.astype(np.int32)), 3, (0, 255, 0))
    cv2.imshow('test', img)
    cv2.waitKey(0)
    '''
    return feat_mean


def normalize(arr: np.array):
    return (arr - np.mean(arr)) / np.std(arr)


def warp_faces_to_features(images, face_features, avg_face):
    warped_and_scaled = np.zeros(shape=(images.shape[0], *IMAGE_SHAPE, 3), dtype=np.uint8)
    mask = np.ones(shape=images.shape[0], dtype=bool)
    for i, (img, ff) in enumerate(zip(tqdm(images, desc='Warping and resizing images'), face_features)):
        mean_positions = get_mean_positions(ff)
        mean_positions['center'] = (mean_positions['left_eye'] + mean_positions['right_eye']) / 2

        f_width, f_height = np.linalg.norm(mean_positions['left_eye'] - mean_positions['right_eye']), \
                            np.linalg.norm(mean_positions['center'] - mean_positions['bottom_lip'])
        f_size = (f_width + f_height) / 2

        if not 0.67 < f_width / f_height < 1.5:
            # Probably not a good match for a face
            mask[i] = False
            continue

        scale_factor = f_size / IDEAL_SIZE
        rotation_angle = np.arctan2(*reversed(mean_positions['right_eye'] - mean_positions['left_eye']))

        rotation_transformation = transform.SimilarityTransform(scale=scale_factor, rotation=rotation_angle,
                                                                translation=mean_positions['center'])

        reposition_transformation = transform.SimilarityTransform(scale=1, rotation=0,
                                                                  translation=tuple(- IDEAL_POS))

        wrp = transform.warp(image=img, inverse_map=(reposition_transformation + rotation_transformation)) \
            [:IMAGE_SHAPE[0], :IMAGE_SHAPE[1]]

        too_black = np.sum(np.sum(wrp, axis=-1) == 0.) > 64*2

        if too_black:
            mask[i] = False
            continue

        warped_and_scaled[i] = wrp
    print(f'Removed {np.sum(~mask)}/{mask.shape[0]}')
    return warped_and_scaled, mask


def colored(img: np.array):
    """Return true if the image is colored. False if grayscale"""
    # Check if image is colored or black and white
    r, g, b = [normalize(img[..., i]) for i in range(3)]
    color_factor = sum([np.mean(np.square(c1 - c2)) for c1, c2 in ((r, g), (r, b), (b, r))])
    return color_factor >= 0.04


if __name__ == '__main__':
    clean_data()
    pass
