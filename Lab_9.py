import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
)


DATA_DIR = Path(__file__).resolve().parent / "data"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SYMBOL_NAMES = {"times": "×"}


def split_symbol_files(data_dir=DATA_DIR, test_fraction=0.3, seed=42):
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Распакуйте папки классов в {data_dir}")

    rng = random.Random(seed)
    train_files, test_files = {}, {}
    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue
        files = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            continue
        if len(files) < 2:
            raise ValueError(f"В {folder} нужно минимум два изображения")
        rng.shuffle(files)
        count = max(1, min(len(files) - 1, round(len(files) * test_fraction)))
        train_files[folder.name] = files[count:]
        test_files[folder.name] = files[:count]

    if not test_files:
        raise ValueError(f"В {data_dir} не найдены изображения в папках классов")
    return train_files, test_files


def read_symbol(path, height=48):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение {path}")

    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate((binary[0], binary[-1], binary[:, 0], binary[:, -1]))
    if np.median(border) < 127:
        binary = cv2.bitwise_not(binary)

    points = cv2.findNonZero(cv2.bitwise_not(binary))
    if points is None:
        raise ValueError(f"Пустое изображение {path}")
    x, y, w, h = cv2.boundingRect(points)
    symbol = binary[y:y + h, x:x + w]
    width = max(1, round(w * height / h))
    return cv2.resize(symbol, (width, height), interpolation=cv2.INTER_NEAREST)


def prepare_symbol(image):
    points = cv2.findNonZero((image > 127).astype(np.uint8))
    if points is None:
        raise ValueError("Пустой фрагмент символа")
    x, y, w, h = cv2.boundingRect(points)
    image = cv2.resize(image[y:y+h, x:x+w], (32, 32))
    return image.astype(np.float32).reshape(-1) / 255


def make_dataset(files_by_class):
    vectors, labels = [], []
    for label, paths in files_by_class.items():
        for path in paths:
            symbol = read_symbol(path)
            symbol = cv2.copyMakeBorder(
                symbol, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=255
            )
            symbol = cv2.erode(symbol, np.ones((3, 3), np.uint8), iterations=1)
            vectors.append(prepare_symbol(cv2.bitwise_not(symbol)))
            labels.append(label)
    return np.asarray(vectors), np.asarray(labels)




train_files, test_files = split_symbol_files()




X_train, y_train = make_dataset(train_files)
X_test, y_test = make_dataset(test_files)




classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(X_train, y_train)




y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))




def generate_sequence(symbol_files, max_symbols=10, space_probability=0.25, seed=None):
    if max_symbols < 1 or not 0 <= space_probability <= 1:
        raise ValueError("max_symbols >= 1, space_probability от 0 до 1")
    if not symbol_files or any(not files for files in symbol_files.values()):
        raise ValueError("Нужны непустые списки изображений классов")

    rng = random.Random(seed)
    count = rng.randint(1, max_symbols)
    classes = sorted(symbol_files)
    images, gaps, text = [], [], []

    for i in range(count):
        label = rng.choice(classes)
        path = rng.choice(symbol_files[label])
        images.append(read_symbol(path))
        if i:
            is_space = rng.random() < space_probability
            gaps.append(30 if is_space else 10)
            if is_space:
                text.append(" ")
        text.append(SYMBOL_NAMES.get(label, label))

    margin = 12
    width = 2 * margin + sum(image.shape[1] for image in images) + sum(gaps)
    canvas = np.full((48 + 2 * margin, width), 255, dtype=np.uint8)
    x = margin
    for i, image in enumerate(images):
        if i:
            x += gaps[i - 1]
        canvas[margin:margin + image.shape[0], x:x + image.shape[1]] = image
        x += image.shape[1]


    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR), "".join(text)

img, expected_text = generate_sequence(test_files, max_symbols=10)
print("Правильная строка:", expected_text)




img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel = np.ones((3, 3), np.uint8)
img_gray = cv2.erode(img_gray, kernel, iterations=1)

img_gray = cv2.bitwise_not(img_gray)

base_size = img_gray.shape[0] + 6, img_gray.shape[1] + 6
base = np.zeros(base_size, dtype=np.uint8)

base[
    3:img_gray.shape[0] + 3,
    3:img_gray.shape[1] + 3
] = img_gray

base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
img_rgb = base

plt.imshow(img_rgb)
plt.show()




ret, thresh = cv2.threshold(img_rgb, 127, 255, 0)
thresh = np.uint8(thresh)

contours, h = cv2.findContours(
    thresh[:, :, 0],
    cv2.RETR_EXTERNAL,
    2,
)

print(len(contours))




crops = []

for i in range(len(contours)):
    cnt = contours[i]
    x, y, w, h = cv2.boundingRect(cnt)

    if w * h > 50:
        cv2.rectangle(
            img_rgb,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2,
        )

        crops.append([x, y, w, h])

print(crops)

plt.imshow(img_rgb)
plt.show()




data_img = []

for i in range(len(crops)):
    x, y, w, h = crops[i]

    x -= 3
    y -= 3


    w += 5
    h += 3

    img = img_gray[y:y+h, x:x+w]


    base_size = img.shape[0] + 2, img.shape[1] + 30
    base = np.zeros(base_size, dtype=np.uint8)

    base[
        1:img.shape[0] + 1,
        15:img.shape[1] + 15
    ] = img

    data_img.append((x, base))

    plt.imshow(base, cmap="gray")
    plt.show()




data_img.sort(key=lambda item: item[0])
symbol_vectors = [prepare_symbol(image) for x, image in data_img]



recognized_text = ""

if symbol_vectors:
    labels = classifier.predict(symbol_vectors)
    ordered_crops = sorted(crops, key=lambda crop: crop[0])

    for i, label in enumerate(labels):
        if i:
            previous_x, _, previous_w, _ = ordered_crops[i - 1]
            gap = ordered_crops[i][0] - (previous_x + previous_w)
            if gap > 20:
                recognized_text += " "
        recognized_text += SYMBOL_NAMES.get(label, label)

print("Правильная строка:", expected_text)
print("Распознано:", recognized_text)
