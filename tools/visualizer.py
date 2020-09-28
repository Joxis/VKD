class Visualizer:

    def __init__(self, g_pids, g_camids, q_pids, q_camids):
        self.g_pids = g_pids
        self.g_camids = g_camids
        self.q_pids = q_pids
        self.q_camids = q_camids

    def run(self, dist_mat):
        num_query, num_gallery = dist_mat.shape
        for i in range(num_query):
            distances = [dist_mat[i, j] for j in range(num_gallery)]
            min_distance = min(distances)

            if min_distance <= 0.17:
                gallery_idx = distances.index(min_distance)
                gallery_id = self.g_pids[gallery_idx]
                gallery_camera_id = self.g_camids[gallery_idx]
                query_id = self.q_pids[i]
                query_camera_id = self.q_camids[i]
                print(
                    "Identity {}.{} is closest ({:.4f}) to identity {}.{}.".format(
                        query_camera_id, query_id, min_distance,
                        gallery_camera_id,
                        gallery_id))
