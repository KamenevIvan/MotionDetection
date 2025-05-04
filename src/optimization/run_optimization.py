import os
import argparse
import time
from datetime import datetime
import sys
from pathlib import Path

from optimization import optimize_parameters

sys.path.append(str(Path(__file__).parent.parent))
import settings

def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Оптимизация параметров детектора с использованием аннотаций')
    
    parser.add_argument('--video', type=str, default=settings.inputfile,
                      help='Путь к видеофайлу')
    parser.add_argument('--annotations', type=str, required=True,
                      help='Путь к файлу с аннотациями')
    parser.add_argument('--output-dir', type=str, default='optimization_results',
                      help='Директория для сохранения результатов')
    parser.add_argument('--trials', type=int, default=settings.OPTIMIZATION_TRIALS,
                      help='Количество испытаний для оптимизации')
    parser.add_argument('--max-frames', type=int, default=None,
                      help='Максимальное количество кадров для оценки')
    
    return parser.parse_args()

def update_settings_file(best_params):
    """Обновляет файл settings.py с новыми параметрами, учитывая условие для buffer_size"""
    settings_path = os.path.join(Path(__file__).parent.parent, 'settings.py')
    
    with open(settings_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    skip = False
    for line in lines:
        if line.startswith('# Оптимизированные параметры'):
            skip = True
        elif skip and line.strip() == '':
            skip = False
        if not skip:
            new_lines.append(line)
    
    new_lines.append('\n# Оптимизированные параметры (автоматическое обновление)\n')
    
    optimized_buffer_size = best_params['buffer_size']
    if optimized_buffer_size < settings.nf_object:
        new_lines.append(f"buffer_size = {settings.nf_object}  # Использовано nf_object, так как оптимизированный размер буфера ({optimized_buffer_size}) был меньше\n")
    else:
        new_lines.append(f"buffer_size = {optimized_buffer_size}\n")

    if best_params['color_space'] == 'HSV':
        new_lines.append(f"color_space = '{best_params['color_space']}'\n")
        new_lines.append(f"threshold_h = {best_params['threshold_h']}\n")
        new_lines.append(f"threshold_s = {best_params['threshold_s']}\n")
        new_lines.append(f"threshold_v = {best_params['threshold_v']}\n")
    elif best_params['color_space'] == 'YCbCr':
        new_lines.append(f"color_space = '{best_params['color_space']}'\n")
        new_lines.append(f"threshold_y = {best_params['threshold_y']}\n")
        new_lines.append(f"threshold_chroma = {best_params['threshold_chroma']}\n")
    else:  # GRAY
        new_lines.append(f"color_space = '{best_params['color_space']}'\n")
        new_lines.append(f"threshold_value = {best_params['threshold']}\n")
    
    new_lines.append(f"blur = {best_params['blur']}\n")
    new_lines.append(f"minimum_are_contours = {best_params['min_area']}\n")
    new_lines.append(f"merge_distance_threshold = {best_params['merge_distance_threshold']}\n")
    new_lines.append(f"overlap_threshold = {best_params['overlap_threshold']}\n")
    
    with open(settings_path, 'w') as f:
        f.writelines(new_lines)

def main():
    args = parse_arguments()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'optimization_{timestamp}')
    
    start_time = time.time()
    
    best_params = optimize_parameters(
        video_path=args.video,
        annotation_path=args.annotations,
        n_trials=args.trials,
        max_frames=args.max_frames,
        output_dir=output_dir
    )
    
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nВремя выполнения: {int(hours)}:{int(minutes):02}:{seconds:.2f}")
    
    update_settings_file(best_params)
    
    print("\nОбновлённые параметры в settings.py:")
    if best_params['color_space'] == 'HSV':
        print(f"  - Цветовое пространство: HSV")
        print(f"  - Порог H: {best_params['threshold_h']}")
        print(f"  - Порог S: {best_params['threshold_s']}")
        print(f"  - Порог V: {best_params['threshold_v']}")
    elif best_params['color_space'] == 'YCbCr':
        print(f"  - Цветовое пространство: YCbCr")
        print(f"  - Порог Y: {best_params['threshold_y']}")
        print(f"  - Порог Chroma: {best_params['threshold_chroma']}")
    else:
        print(f"  - Цветовое пространство: GRAY")
        print(f"  - Порог: {best_params['threshold']}")
    
    print(f"  - Размытие: {best_params['blur']}")
    print(f"  - Минимальная площадь: {best_params['min_area']}")
    print(f"  - Размер буфера: {best_params['buffer_size']}")
    print(f"  - Порог расстояния для объединения: {best_params['merge_distance_threshold']}")
    print(f"  - Порог перекрытия: {best_params['overlap_threshold']}")

if __name__ == '__main__':
    main()