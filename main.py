import cv2
import numpy as np
from pathlib import Path

# Папки проєкту
for folder in ["images", "videos", "results"]:
    Path(folder).mkdir(exist_ok=True)

# Мінімальна площа фігури в пікселях
MIN_AREA = 500


def detect_shapes(frame):
    """
    Пошук геометричних фігур на зображенні.
    Найкраще працює з темними фігурами
    на світлому однорідному фоні.
    """

    result = frame.copy()

    # Перетворення на відтінки сірого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Зменшення шуму
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Перетворення на чорно-біле зображення
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Пошук зовнішніх контурів
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    counts = {
        "Triangle": 0,
        "Square": 0,
        "Rectangle": 0,
        "Pentagon": 0,
        "Circle": 0
    }

    for contour in contours:
        area = cv2.contourArea(contour)

        # Пропускаємо дуже маленькі об'єкти
        if area < MIN_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        # Спрощення контуру до багатокутника
        approximation = cv2.approxPolyDP(
            contour,
            0.025 * perimeter,
            True
        )

        vertices = len(approximation)

        # Прямокутник навколо об'єкта
        x, y, w, h = cv2.boundingRect(approximation)

        # Круглість: для ідеального кола близька до 1
        circularity = (
            4 * np.pi * area / (perimeter ** 2)
        )

        # Визначення типу фігури
        if vertices == 3:
            shape = "Triangle"

        elif vertices == 4:
            # Перевіряємо, чи кути близькі до 90 градусів
            points = approximation.reshape(4, 2).astype(float)
            angles_ok = True

            for i in range(4):
                previous = points[(i - 1) % 4] - points[i]
                following = points[(i + 1) % 4] - points[i]

                denominator = (
                    np.linalg.norm(previous)
                    * np.linalg.norm(following)
                )

                if denominator == 0:
                    angles_ok = False
                    break

                cosine = np.dot(
                    previous, following
                ) / denominator

                if abs(cosine) > 0.25:
                    angles_ok = False
                    break

            if not angles_ok:
                continue

            # Порівняння довжин сусідніх сторін
            side1 = np.linalg.norm(points[1] - points[0])
            side2 = np.linalg.norm(points[2] - points[1])

            if min(side1, side2) == 0:
                continue

            ratio = max(side1, side2) / min(side1, side2)

            if ratio <= 1.15:
                shape = "Square"
            else:
                shape = "Rectangle"

        elif vertices == 5:
            shape = "Pentagon"

        elif vertices >= 6 and circularity >= 0.75:
            shape = "Circle"

        else:
            continue

        counts[shape] += 1

        # Малюємо контур зеленим кольором
        cv2.drawContours(
            result,
            [contour],
            -1,
            (0, 255, 0),
            3
        )

        # Підписуємо назву фігури
        cv2.putText(
            result,
            shape,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    # Загальна кількість фігур
    total = sum(counts.values())

    # Інформаційна панель
    cv2.rectangle(
        result,
        (0, 0),
        (result.shape[1], 48),
        (35, 35, 35),
        -1
    )

    cv2.putText(
        result,
        f"Detected shapes: {total}",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    return result, counts


def print_counts(counts):
    """Виведення кількості фігур."""
    print("\n--- РЕЗУЛЬТАТИ РОЗПІЗНАВАННЯ ---")

    for name, count in counts.items():
        print(f"{name}: {count}")

    print(f"Всього: {sum(counts.values())}")


def process_image():
    """Розпізнавання фігур на фотографії."""
    path = input("Введіть шлях до зображення: ").strip('" ')

    # imdecode підтримує шляхи з кирилицею
    try:
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        image = None

    if image is None:
        print("Помилка: не вдалося відкрити зображення.")
        return

    result, counts = detect_shapes(image)

    output = Path("results/image_result.png")

    success, encoded = cv2.imencode(".png", result)

    if success:
        encoded.tofile(str(output))
        print(f"Результат збережено: {output}")

    print_counts(counts)

    cv2.imshow("Original image", image)
    cv2.imshow("Recognized shapes", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_camera():
    """Розпізнавання фігур через вебкамеру."""
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("Помилка: вебкамеру не знайдено.")
        return

    print("\nКамера запущена.")
    print("S - зберегти результат")
    print("Q - вийти")

    while True:
        success, frame = camera.read()

        if not success:
            print("Помилка отримання кадру.")
            break

        # Дзеркальне відображення
        frame = cv2.flip(frame, 1)

        result, counts = detect_shapes(frame)

        cv2.imshow("Shape recognition - Camera", result)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            output = "results/camera_result.png"
            cv2.imwrite(output, result)
            print(f"Збережено: {output}")
            print_counts(counts)

        elif key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


def process_video():
    """Розпізнавання фігур у відеофайлі."""
    path = input("Введіть шлях до відео: ").strip('" ')

    video = cv2.VideoCapture(path)

    if not video.isOpened():
        print("Помилка: не вдалося відкрити відео.")
        return

    fps = video.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output = "results/video_result.mp4"

    writer = cv2.VideoWriter(
        output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        print("Помилка створення вихідного відео.")
        video.release()
        return

    print("Обробка відео. Q - завершити.")

    while True:
        success, frame = video.read()

        if not success:
            break

        result, counts = detect_shapes(frame)

        writer.write(result)

        cv2.imshow("Shape recognition - Video", result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Результат збережено: {output}")


def main():
    """Головне меню програми."""
    while True:
        print("\n" + "=" * 45)
        print("СИСТЕМА РОЗПІЗНАВАННЯ ГЕОМЕТРИЧНИХ ФІГУР")
        print("Бібліотека: OpenCV")
        print("=" * 45)

        print("1. Розпізнавання з фотографії")
        print("2. Розпізнавання через вебкамеру")
        print("3. Розпізнавання з відеофайлу")
        print("0. Вийти")

        choice = input("\nОберіть дію: ").strip()

        if choice == "1":
            process_image()

        elif choice == "2":
            process_camera()

        elif choice == "3":
            process_video()

        elif choice == "0":
            print("Програму завершено.")
            break

        else:
            print("Помилка! Виберіть пункт меню.")


if __name__ == "__main__":
    main()