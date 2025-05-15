import os
import cv2
import shutil
from tqdm import tqdm  # для красивого прогресс-бара (установите через pip install tqdm)

def create_video_from_images(img_folder, output_folder, fps=30):
    # Получаем список изображений в папке img
    images = [img for img in os.listdir(img_folder) if img.endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    if not images:
        print(f"Нет изображений в папке {img_folder}. Пропускаю.")
        return

    # Сортируем изображения по имени (важно, если они нумерованы)
    images.sort()

    # Определяем размер первого изображения (предполагаем, что все имеют одинаковый размер)
    first_image = cv2.imread(os.path.join(img_folder, images[0]))
    height, width, _ = first_image.shape

    # Создаём имя видео (название родительской папки + .mp4)
    video_name = os.path.basename(os.path.dirname(img_folder)) + ".mp4"
    video_path = os.path.join(output_folder, video_name)

    # Создаём объект VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # кодек для MP4
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Добавляем кадры в видео
    for image in tqdm(images, desc=f"Создание видео {video_name}"):
        img_path = os.path.join(img_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
    print(f"Видео сохранено: {video_path}")

def process_all_folders(root_folder="TLP"):
    for root, dirs, files in os.walk(root_folder):
        if "img" in dirs:
            img_folder = os.path.join(root, "img")
            parent_folder = root  # папка, в которой лежит img
            print(f"\nОбработка папки: {parent_folder}")

            # Создаём видео
            create_video_from_images(img_folder, parent_folder)

            # Удаляем папку img
            shutil.rmtree(img_folder)
            print(f"Папка img удалена: {img_folder}")

if __name__ == "__main__":
    process_all_folders("TLP")  # укажите путь, если TLP не в той же папке
    print("\nГотово! Все видео созданы, папки img удалены.")