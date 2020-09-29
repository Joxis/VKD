import os
import shutil
import os.path as osp
import errno

import numpy as np
import cv2 as cv
from tqdm import tqdm
from torchvision.utils import save_image

GRID_SPACING = 10
QUERY_EXTRA_SPACING = 90
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
        distmat, dataset, data_type, width=128, height=256, save_dir='', topk=10
):
    """Visualizes ranked results.
    Supports both image-reid and video-reid.
    For image-reid, ranks will be plotted in a single figure. For video-reid,
    ranks will be saved in folders each containing a tracklet.
    Args:
        distmat (numpy.ndarray): distance matrix of shape
            (num_query, num_gallery).
        dataset (tuple): a 2-tuple containing (query, gallery), each of which
            contains tuples of (img_path(s), pid, camid, dsetid).
        data_type (str): "image" or "video".
        width (int, optional): resized image width. Default is 128.
        height (int, optional): resized image height. Default is 256.
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be
            visualized. Default is 10.
    """
    num_q, num_g = distmat.shape
    mkdir_if_missing(save_dir)

    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Visualizing top-{} ranks ...'.format(topk))

    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)

    indices = np.argsort(distmat, axis=1)

    def _cp_img_to(src, dst, rank, prefix, matched=False):
        """
        Args:
            src: image path or tuple (for vidreid)
            dst: target directory
            rank: int, denoting ranked position, starting from 1
            prefix: string
            matched: bool
        """
        if isinstance(src, (tuple, list)):
            if prefix == 'gallery':
                suffix = 'TRUE' if matched else 'FALSE'
                dst = osp.join(
                    dst, prefix + '_top' + str(rank).zfill(3)
                ) + '_' + suffix
            else:
                dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            mkdir_if_missing(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(
                dst, prefix + '_top' + str(rank).zfill(3) + '_name_' +
                     osp.basename(src)
            )
            shutil.copy(src, dst)

    for q_idx in range(num_q):
        qimg_path, qpid, qcamid = query[q_idx][:3]
        qimg_path_name = qimg_path[0] if isinstance(
            qimg_path, (tuple, list)
        ) else qimg_path

        if data_type == 'image':
            qimg = cv.imread(qimg_path)
            qimg = cv.resize(qimg, (width, height))
            qimg = cv.copyMakeBorder(
                qimg, BW, BW, BW, BW, cv.BORDER_CONSTANT, value=(0, 0, 0)
            )
            # resize twice to ensure that the border width is consistent across
            #   images
            qimg = cv.resize(qimg, (width, height))
            num_cols = topk + 1
            grid_img = 255 * np.ones(
                (
                    height,
                    num_cols * width + topk * GRID_SPACING + QUERY_EXTRA_SPACING,
                    3
                ),
                dtype=np.uint8
            )
            grid_img[:, :width, :] = qimg
        else:
            qdir = osp.join(
                save_dir, osp.basename(osp.splitext(qimg_path_name)[0])
            )
            mkdir_if_missing(qdir)
            _cp_img_to(qimg_path, qdir, rank=0, prefix='query')

        rank_idx = 1
        for g_idx in indices[q_idx, :]:
            gimg_path, gpid, gcamid = gallery[g_idx][:3]
            invalid = (qpid == gpid) & (qcamid == gcamid)

            if not invalid:
                matched = gpid == qpid
                if data_type == 'image':
                    border_color = GREEN if matched else RED
                    gimg = cv.imread(gimg_path)
                    gimg = cv.resize(gimg, (width, height))
                    gimg = cv.copyMakeBorder(
                        gimg,
                        BW,
                        BW,
                        BW,
                        BW,
                        cv.BORDER_CONSTANT,
                        value=border_color
                    )
                    gimg = cv.resize(gimg, (width, height))
                    start = rank_idx * width + rank_idx * GRID_SPACING + QUERY_EXTRA_SPACING
                    end = (
                                  rank_idx + 1
                          ) * width + rank_idx * GRID_SPACING + QUERY_EXTRA_SPACING
                    grid_img[:, start:end, :] = gimg
                else:
                    _cp_img_to(
                        gimg_path,
                        qdir,
                        rank=rank_idx,
                        prefix='gallery',
                        matched=matched
                    )

                rank_idx += 1
                if rank_idx > topk:
                    break

        if data_type == 'image':
            imname = osp.basename(osp.splitext(qimg_path_name)[0])
            cv.imwrite(osp.join(save_dir, imname + '.jpg'), grid_img)

        if (q_idx + 1) % 100 == 0:
            print('- done {}/{}'.format(q_idx + 1, num_q))

    print('Done. Images have been saved to "{}" ...'.format(save_dir))


class Visualizer:

    def __init__(self, g_pids, g_camids, q_pids, q_camids, g_dataset, q_dataset,
                 out_dir='output/', k=10):
        # self.g_pids = g_pids
        # self.g_camids = g_camids
        # self.q_pids = q_pids
        # self.q_camids = q_camids
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
        # cv.imwrite(image_path, image)
        save_image(image, image_path)

    @staticmethod
    def save_query_images(images_dir, image_tuple):
        images, p_id, cam_id = image_tuple
        for i, image in enumerate(images):
            image_path = os.path.join(
                images_dir, "q-{}-{}_c{}.jpg".format(str(i).zfill(2), p_id,
                                                     str(cam_id).zfill(2)))
            Visualizer.save_image(image, image_path)

    @staticmethod
    def save_gallery_images(images_dir, image_tuples, distances):
        for i, (images, p_id, cam_id) in enumerate(image_tuples):
            image_path = os.path.join(
                images_dir, "g-{:.2f}-{}_c{}.jpg".format(distances[i], p_id,
                                                         str(cam_id).zfill(2)))
            Visualizer.save_image(images[0], image_path)

    def run(self, dist_mat, num=100):
        num_query, num_gallery = dist_mat.shape

        print(len(self.q_dataset.dataset), num_query,
              len(self.g_dataset.dataset), num_gallery)
        print(self.q_dataset.dataset[0])
        # print(self.q_dataset[0][0].shape, self.q_dataset[0][1:])
        # print(self.g_dataset[0][0].shape, self.g_dataset[0][1:])

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

            self.save_query_images(out_pid_dir, self.q_dataset[i])
            self.save_gallery_images(out_pid_dir, result_image_tuples,
                                     result_distances)
