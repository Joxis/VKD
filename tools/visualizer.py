import os
import os.path as osp
import errno

import numpy as np
import cv2 as cv
from tqdm import tqdm

GRID_SPACING = 10
Q_SPACING = 90
BW = 5  # border width
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def visualize_ranked_results(
        distmat, g_dataset, q_dataset, width=128, height=256,
        save_dir='', topk=10
):
    num_q, num_g = distmat.shape
    mkdir_if_missing(save_dir)

    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Visualizing top-{} ranks ...'.format(topk))

    query, gallery = q_dataset, g_dataset
    assert num_q == len(query)
    assert num_g == len(gallery)

    indices = np.argsort(distmat, axis=1)

    for q_idx in range(num_q):
        qimg_paths, qpid, qcamid = query[q_idx]
        qimg_path = qimg_paths[0]
        qimg_path_name = qimg_path[0] if isinstance(
            qimg_path, (tuple, list)
        ) else qimg_path

        qimg = cv.imread(qimg_path)
        qimg = cv.resize(qimg, (width, height))
        qimg = cv.copyMakeBorder(
            qimg, BW, BW, BW, BW, cv.BORDER_CONSTANT, value=(0, 0, 0)
        )
        # resize twice to ensure that the border width is consistent
        qimg = cv.resize(qimg, (width, height))
        num_cols = topk + 1
        grid_img = 255 * np.ones(
            (
                height,
                num_cols * width + topk * GRID_SPACING + Q_SPACING,
                3
            ),
            dtype=np.uint8
        )
        grid_img[:, :width, :] = qimg

        rank_idx = 1
        for g_idx in indices[q_idx, :]:
            gimg_paths, gpid, gcamid = gallery[g_idx]
            gimg_path = gimg_paths[0]
            invalid = (qpid == gpid) & (qcamid == gcamid)

            if not invalid:
                matched = gpid == qpid
                border_color = GREEN if matched else RED
                gimg = cv.imread(gimg_path)
                gimg = cv.resize(gimg, (width, height))
                gimg = cv.copyMakeBorder(
                    gimg, BW, BW, BW, BW, cv.BORDER_CONSTANT, value=border_color
                )
                gimg = cv.resize(gimg, (width, height))
                start = rank_idx * width + rank_idx * GRID_SPACING + Q_SPACING
                end = (
                              rank_idx + 1
                      ) * width + rank_idx * GRID_SPACING + Q_SPACING
                grid_img[:, start:end, :] = gimg

                rank_idx += 1
                if rank_idx > topk:
                    break

        imname = osp.basename(osp.splitext(qimg_path_name)[0])
        cv.imwrite(osp.join(save_dir, imname + '.jpg'), grid_img)

        if (q_idx + 1) % 100 == 0:
            print('- done {}/{}'.format(q_idx + 1, num_q))

    print('Done. Images have been saved to "{}" ...'.format(save_dir))


class Visualizer:

    def __init__(self, g_dataset, q_dataset,
                 out_dir='output/', k=10):
        self.g_dataset = g_dataset
        self.q_dataset = q_dataset
        self.out_dir = out_dir
        self.k = k

        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)

    @staticmethod
    def save_image(image, image_path):
        # image = image.numpy().transpose((1, 2, 0)) * 255
        # image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        cv.imwrite(image_path, image)

    @staticmethod
    def read_image(image_path):
        return cv.imread(image_path)

    @staticmethod
    def save_query_images(images_dir, image_tuple):
        images, p_id, cam_id = image_tuple
        for i, image_path in enumerate(images):
            image = Visualizer.read_image(image_path)
            out_image_path = os.path.join(
                images_dir, "q-{}-{}_c{}.jpg".format(str(i).zfill(2), p_id,
                                                     str(cam_id).zfill(2)))
            Visualizer.save_image(image, out_image_path)

    @staticmethod
    def save_gallery_images(images_dir, image_tuples, distances):
        for i, (image_paths, p_id, cam_id) in enumerate(image_tuples):
            out_image_path = os.path.join(
                images_dir, "g-{:.2f}-{}_c{}.jpg".format(distances[i], p_id,
                                                         str(cam_id).zfill(2)))

            image = Visualizer.read_image(image_paths[0])
            Visualizer.save_image(image, out_image_path)

    def run(self, dist_mat, num=100):
        num_query, num_gallery = dist_mat.shape

        print("Saving images...")
        for i in tqdm(range(min(num_query, num))):
            # Get the k closest identities
            distances = np.array([dist_mat[i, j] for j in range(num_gallery)])
            idx = np.argpartition(distances, self.k)[:self.k]
            min_k = idx[np.argsort(distances[idx])]

            # Create a directory to store the images of the top k matches
            query_pid, query_cam_id = self.q_dataset[i][1:]
            out_pid_dir = os.path.join(
                self.out_dir, "{}_c{}".format(str(query_pid).zfill(6),
                                              str(query_cam_id).zfill(2)))
            if not os.path.isdir(out_pid_dir):
                os.makedirs(out_pid_dir)

            result_image_tuples = []
            result_distances = []
            for min_idx in min_k:
                result_distances.append(distances[min_idx])
                result_image_tuples.append(self.g_dataset[min_idx])

            # Save images
            self.save_query_images(out_pid_dir, self.q_dataset[i])
            self.save_gallery_images(out_pid_dir, result_image_tuples,
                                     result_distances)
            visualize_ranked_results(dist_mat, self.g_dataset, self.q_dataset,
                                     save_dir=out_pid_dir)
