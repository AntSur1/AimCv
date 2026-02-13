import numpy as np

P0 = np.array([-0.13736473,  0.13483921,  0.72674544])
P1 = np.array([-0.08077869, -0.03822985,  0.96909398])
P2 = np.array([-0.17935716, -0.03354962,  1.00529492])

rear_mid = 0.5 * (P1 + P2)
aim_vec = rear_mid - P0
aim_dist = np.linalg.norm(aim_vec)
aim_dir = aim_vec / aim_dist

def compute_offsets(P):
    vec = P - P0
    longitudinal = np.dot(vec, aim_dir)
    lateral_vec = vec - longitudinal * aim_dir
    lateral_mag = np.linalg.norm(lateral_vec)
    return longitudinal, lateral_vec, lateral_mag

for i, P in enumerate([P1, P2], start=1):
    long, lat_vec, lat_mag = compute_offsets(P)
    print(f"Marker {i}:")
    print("  Longitudinal (along barrel):", long)
    print("  Lateral vector:", lat_vec)
    print("  Lateral magnitude:", lat_mag)

print("Rear midpoint distance from front:", np.linalg.norm(rear_mid - P0))

def compute_gun_frame_points(P0, P1, P2):
    """
    Compute gun-frame coordinates Zg, Lg, Rg from 3D marker positions.

    Parameters
    ----------
    P0 : np.array
        Front marker (Z), shape (3,)
    P1 : np.array
        Rear marker 1, shape (3,)
    P2 : np.array
        Rear marker 2, shape (3,)

    Returns
    -------
    Zg, Lg, Rg : np.array
        Gun-frame points in meters:
        Zg = front marker at origin
        Lg = left rear marker
        Rg = right rear marker
    """
    # Compute rear midpoint and aim direction
    rear_mid = 0.5 * (P1 + P2)
    aim_vec = rear_mid - P0
    aim_dir = aim_vec / np.linalg.norm(aim_vec)

    # Function to compute longitudinal and lateral components
    def to_gun_frame(P):
        vec = P - P0
        longitudinal = np.dot(vec, aim_dir)
        lateral_vec = vec - longitudinal * aim_dir
        return longitudinal, lateral_vec

    # Compute offsets for rear markers
    long1, lat1 = to_gun_frame(P1)
    long2, lat2 = to_gun_frame(P2)

    # Build gun-frame points
    Zg = np.array([0.0, 0.0, 0.0])  # front marker at origin
    Lg = np.array([lat1[0], lat1[1], -long1])  # X = lateral X, Y = lateral Y, Z = -longitudinal
    Rg = np.array([lat2[0], lat2[1], -long2])

    return Zg, Lg, Rg

Zg, Lg, Rg = compute_gun_frame_points(P0, P1, P2)

print("Zg =", Zg)
print("Lg =", Lg)
print("Rg =", Rg)

