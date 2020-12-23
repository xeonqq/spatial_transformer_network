import matplotlib.pyplot as plt
from math import ceil


def draw_samples(images, title, images_per_row=5, save_fig=False):
    num = len(images)
    per_row = min(images_per_row, num)
    rows = ceil(num / per_row)
    fig, axs = plt.subplots(rows, per_row)
    count = 0
    for i in range(rows):

        for j in range(images_per_row):
            count += 1
            if count > num:
                break
            if rows == 1:
                axs[j + i * per_row].imshow(images[j + i * per_row], cmap="gray")
            else:
                axs[i, j].imshow(images[j + i * per_row], cmap="gray")
    fig.suptitle(title)
    if save_fig:
        print(title + ".png generated")
        plt.savefig(title + "png")
    plt.show()
