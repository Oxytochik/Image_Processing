import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


image = cv.imread('periodic_noise.JPG', cv.IMREAD_GRAYSCALE)

h, w = image.shape
s = min(h, w)

image = cv.resize(
    image[(h-s)//2:(h-s)//2+s, (w-s)//2:(w-s)//2+s],
    (512, 512)
)


def haar(image, levels):
    c = image.astype(np.float32)

    for level in range(levels):
        n = c.shape[0] >> level
        b = c[:n, :n]

        b = np.hstack((
            (b[:, 0::2] + b[:, 1::2]) / 2,
            (b[:, 0::2] - b[:, 1::2]) / 2
        ))

        c[:n, :n] = np.vstack((
            (b[0::2] + b[1::2]) / 2,
            (b[0::2] - b[1::2]) / 2
        ))

    return c


def ihaar(coef, levels):
    c = coef.copy()

    for level in range(levels - 1, -1, -1):
        n = c.shape[0] >> level
        k = n // 2
        b = c[:n, :n]
        t = np.empty_like(b)

        t[0::2] = b[:k] + b[k:]
        t[1::2] = b[:k] - b[k:]

        c[:n, :n][:, 0::2] = t[:, :k] + t[:, k:]
        c[:n, :n][:, 1::2] = t[:, :k] - t[:, k:]

    return c


plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.title('Исходное 512×512')
plt.axis('off')

for level in range(1, 4):
    coef = haar(image, level)
    n = 512 >> level

    # Усиление деталей только для отображения.
    vis = np.clip(np.abs(coef) * 5, 0, 255)
    vis[:n, :n] = coef[:n, :n]

    plt.subplot(1, 4, level + 1)
    plt.imshow(vis, cmap='gray', vmin=0, vmax=255)
    plt.title(f'{level} ступень, LL {n}×{n}')
    plt.axis('off')

plt.tight_layout()
plt.show()


coef = haar(image, 3)
print('Ошибка восстановления:', np.abs(ihaar(coef, 3) - image).max())


#
compressed = coef.copy()
compressed[np.abs(compressed) < 5] = 0

compressed[:64, :64] = coef[:64, :64]

restored = np.clip(ihaar(compressed, 3), 0, 255)

print('Нулевых коэффициентов:',
      round(np.mean(compressed == 0) * 100, 1), '%')
print('MSE:', round(np.mean((restored - image) ** 2), 2))


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.title('Исходное')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(restored, cmap='gray', vmin=0, vmax=255)
plt.title('После обнуления слабых деталей')
plt.axis('off')

plt.tight_layout()
plt.show()