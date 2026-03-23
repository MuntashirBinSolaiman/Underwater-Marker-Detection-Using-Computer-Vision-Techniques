import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    def show_mask(self, mask, axis, obj_id=None, random_color=False):
        if random_color:
            random_rgb = np.random.random(3)
            alpha = np.array([0.6])
            color = np.concatenate([random_rgb, alpha], axis=0)
        else:
            colormap = plt.get_cmap("tab10")
            color_index = 0 if obj_id is None else obj_id
            rgb = colormap(color_index)[:3]
            alpha = 0.6
            color = np.array([*rgb, alpha])

        height = mask.shape[-2]
        width = mask.shape[-1]
        reshaped_mask = mask.reshape(height, width, 1)
        reshaped_color = color.reshape(1, 1, -1)
        mask_image = reshaped_mask * reshaped_color
        axis.imshow(mask_image)

    def show_points(self, coordinates, labels, axis, marker_size=200):
        positive_points = coordinates[labels == 1]
        negative_points = coordinates[labels == 0]

        axis.scatter(positive_points[:, 0], positive_points[:, 1], color='green', marker='*',
                     s=marker_size, edgecolor='white', linewidth=1.25)
        axis.scatter(negative_points[:, 0], negative_points[:, 1], color='red', marker='*',
                     s=marker_size, edgecolor='white', linewidth=1.25)

    def show_box(self, box, axis):
        x0, y0 = box[0], box[1]
        width = box[2] - x0
        height = box[3] - y0
        axis.add_patch(plt.Rectangle((x0, y0), width, height,
                                     edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
