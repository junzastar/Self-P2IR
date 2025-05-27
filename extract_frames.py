import numpy as np
from scipy.spatial.transform import quaternion_from_rotation_matrix

def calculate_rre(gt_rotation, estimated_rotation):
    # Convert rotation matrices to quaternions
    gt_quaternion = quaternion_from_rotation_matrix(gt_rotation)
    estimated_quaternion = quaternion_from_rotation_matrix(estimated_rotation)
    
    # Calculate the angular difference between the quaternions
    angular_difference = np.arccos(2 * np.dot(gt_quaternion, estimated_quaternion)**2 - 1)
    
    # Convert the angular difference to degrees
    rre = np.degrees(angular_difference)
    
    return rre

def calculate_rte(gt_translation, estimated_translation):
    # Calculate the Euclidean distance between the translations
    rte = np.linalg.norm(gt_translation - estimated_translation)
    
    return rte

# Example usage
gt_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Ground truth rotation matrix
estimated_rotation = np.array([[0.999, 0.001, 0], [-0.001, 0.999, 0], [0, 0, 1]])  # Estimated rotation matrix
gt_translation = np.array([1, 2, 3])  # Ground truth translation vector
estimated_translation = np.array([1.1, 2.2, 2.9])  # Estimated translation vector

rre = calculate_rre(gt_rotation, estimated_rotation)
rte = calculate_rte(gt_translation, estimated_translation)

print("RRE:", rre)
print("RTE:", rte)
