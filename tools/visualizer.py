import os

import numpy as np


class Visualizer:

    def __init__(self, g_pids, g_camids, q_pids, q_camids, out_dir='output/',
                 k=10):
        self.g_pids = g_pids
        self.g_camids = g_camids
        self.q_pids = q_pids
        self.q_camids = q_camids
        self.out_dir = out_dir
        self.k = k

    # def get_images(self, pid):
    #     # TODO
    #     return []
    #
    # @staticmethod
    # def save_query_images(images_dir, images):
    #     for i, image in enumerate(images):
    #         cv.imwrite(os.path.join(images_dir, "query{}.jpg".format(i)), image)

    def run(self, dist_mat):
        num_query, num_gallery = dist_mat.shape
        for i in range(num_query):
            # Get the k closest identities
            distances = np.array([dist_mat[i, j] for j in range(num_gallery)])
            idx = np.argpartition(distances, self.k)[:self.k]
            min_k = idx[np.argsort(distances[idx])]

            # Create a directory to store the images of the top k matches
            query_id = self.q_pids[i]
            query_camera_id = self.q_camids[i]
            out_pid_dir = os.path.join(self.out_dir, str(query_id).zfill(8))
            if not os.path.isdir(out_pid_dir):
                os.makedirs(out_pid_dir)

            # query_images = self.get_images(query_id)
            # self.save_query_images(out_pid_dir, query_images)

            print("==== {}_{}".format(query_camera_id, query_id))
            for min_idx in min_k:
                distance = distances[min_idx]
                gallery_id = self.g_pids[min_idx]
                gallery_camera_id = self.g_camids[min_idx]
                print("\t{}_{} ({:.2f})".format(gallery_camera_id, gallery_id,
                                                distance))
