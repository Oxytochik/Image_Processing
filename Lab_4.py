import numpy as np
import cv2
import matplotlib.pyplot as plt


image = cv2.imread('sar_1.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


plt.figure(figsize=(5, 4))
plt.title("Оригинал", fontsize=8)
plt.imshow(image_gray, cmap='gray')
plt.show()


def homo_average(img, mask, point, T):
    av_val = img[mask > 0].sum() / np.count_nonzero(img[mask > 0])
    if abs(av_val - img[point]) <= T:
        return True


def region_growing(image, seed_point, homo_fun, r, T):
    mask = np.zeros(image.shape, np.uint8)
    mask[seed_point] = 1
    count = 1
    while count > 0:
        count = 0
        local_mask = np.zeros(image.shape, np.uint8)
        for i in range(r, image.shape[0] - r):
            for j in range(r, image.shape[1] - r):
                if mask[i, j] == 0 and mask[i - r:i + r, j - r: j + r].sum() > 0:
                    if homo_fun(image, mask, (i, j), T):
                        local_mask[i, j] = 1
        count = np.count_nonzero(local_mask)
        print(count)
        mask += local_mask
    return mask * 255


seed_point = (250, 250)
mask = region_growing(image_gray, seed_point, homo_average, 2, 20)

plt.figure(figsize=(6, 5))
plt.title("Результат region growing", fontsize=9)
plt.imshow(mask, cmap='gray')
plt.show()



def homo_local_median(img, mask, point, T):
    i, j = point
    patch = img[max(0, i-1):i+2, max(0, j-1):j+2]
    return abs(float(img[i, j]) - float(np.median(patch))) <= T

seed_cmp = (160, 132)


mask_avg  = region_growing(image_gray, seed_cmp, homo_average, 2, 20)
mask_lmed = region_growing(image_gray, seed_cmp, homo_local_median, 1, 5)


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image_gray, cmap='gray')
plt.plot(seed_cmp[1], seed_cmp[0], 'r*', markersize=15)
plt.title('Оригинал + seed')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask_avg, cmap='gray')
plt.title(f'homo_average (T=20)\n{np.count_nonzero(mask_avg)} px')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(mask_lmed, cmap='gray')
plt.title(f'homo_local_median (T=12)\n{np.count_nonzero(mask_lmed)} px')
plt.axis('off')
plt.tight_layout()
plt.show()

palms = cv2.imread('palm_1.JPG')
palms_gray = cv2.cvtColor(palms, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(palms_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

FG_RATIO = 0.15
_, sure_fg = cv2.threshold(dist, FG_RATIO * dist.max(), 255, 0)
sure_fg = np.uint8(sure_fg)


n, lbl, stats, _ = cv2.connectedComponentsWithStats(sure_fg, connectivity=8)
clean_fg = np.zeros_like(sure_fg)
for i in range(1, n):
    if stats[i, cv2.CC_STAT_AREA] > 150:
        clean_fg[lbl == i] = 255
sure_fg = clean_fg

sure_bg = cv2.dilate(opening, kernel, iterations=3)

unknown = cv2.subtract(sure_bg, sure_fg)

_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

markers = cv2.watershed(palms, markers)


num_trees = len([l for l in np.unique(markers) if l not in (-1, 0, 1)])

palms_vis = palms.copy()
palms_vis[markers == -1] = [0, 0, 255]

plt.figure(figsize=(18, 5))

plt.subplot(1, 4, 1)
plt.imshow(palms_gray, cmap='gray')
plt.title('1. Оригинал')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(dist, cmap='gray')
plt.title('2. Distance Transform')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(sure_fg, cmap='gray')
plt.title('3. Центры (после фильтра площади)')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(cv2.cvtColor(palms_vis, cv2.COLOR_BGR2RGB))
plt.title(f'4. Watershed: {num_trees} пальм')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"===> Итоговое количество найденных пальм: {num_trees}")