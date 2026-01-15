# visualize /media/vision/lzy/image_zyl/3EED/vis_results/drone/drone_Outdoor_Day_penno_parking_2_000311_iou_0.46_visualization_bboxes.ply

import numpy as np
path = '/media/vision/lzy/image_zyl/3EED/vis_results/drone/drone_Outdoor_Day_penno_parking_2_000311_iou_0.46_visualization_bboxes.ply'
point_cloud = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
print("Point cloud shape:", point_cloud.shape)