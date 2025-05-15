import os
import time
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import subprocess
import shutil
from collections import defaultdict
from datetime import datetime

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Глобальное хранилище статусов обработки
processing_tasks = defaultdict(dict)
task_counter = 0
task_lock = threading.Lock()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_task_info():
    """Возвращает информацию о текущих задачах"""
    with task_lock:
        active_tasks = sum(1 for task in processing_tasks.values() if task.get('is_processing', False))
        waiting_tasks = sum(1 for task in processing_tasks.values() if not task.get('is_processing', False) and not task.get('completed', False))
        completed_tasks = sum(1 for task in processing_tasks.values() if task.get('completed', False))
        
        return {
            'active': active_tasks,
            'waiting': waiting_tasks,
            'completed': completed_tasks,
            'total': len(processing_tasks)
        }

def process_video(task_id, input_path, output_path):
    """Полная функция обработки видео с детекцией движения"""
    try:
        # Инициализация информации о задаче
        with task_lock:
            processing_tasks[task_id].update({
                'is_processing': True,
                'start_time': datetime.now().strftime("%H:%M:%S"),
                'progress': 0,
                'status': 'Initializing',
                'current_frame': 0,
                'total_frames': 0,
                'processing_speed': 0
            })

        # Генерируем уникальные имена для выходных файлов
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        timestamp = int(time.time())
        
        output_json = os.path.join(app.config['PROCESSED_FOLDER'], 
                                 f"{base_name}_{timestamp}_detections.json")
        output_video = os.path.join(app.config['PROCESSED_FOLDER'], 
                                  f"{base_name}_{timestamp}_processed.mp4")

        # Убедимся что папка существует
        os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

        # Команда для запуска обработчика
        cmd = [
            'python', 'main.py',
            '--input', input_path,
            '--output', output_json,
            '--video', output_video
        ]

        # Запускаем процесс с выводом в реальном времени
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
            text=True
        )


        # Переменные для парсинга прогресса
        total_frames = 0
        current_frame = 0
        processing_speed = 0

        # Читаем вывод процесса в реальном времени
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break

            if output:
                # Парсим общее количество кадров
                if "Total frames:" in output:
                    try:
                        total_frames = int(output.split("Total frames:")[1].strip())
                        with task_lock:
                            processing_tasks[task_id]['total_frames'] = total_frames
                    except (ValueError, IndexError):
                        pass

                # Парсим строку прогресса
                elif "Progress:" in output:
                    try:
                        # Пример строки: "Progress: 25.5% | Frame: 255/1000 | Speed: 30.1fps"
                        parts = output.split('|')
                        
                        # Получаем процент выполнения
                        percent_part = parts[0]
                        percent = float(percent_part.split("Progress:")[1].split("%")[0].strip())
                        
                        # Получаем номер текущего кадра
                        frame_part = parts[1]
                        current_frame = int(frame_part.split("Frame:")[1].split("/")[0].strip())
                        
                        # Получаем скорость обработки
                        speed_part = parts[2]
                        processing_speed = float(speed_part.split("Speed:")[1].split("fps")[0].strip())

                        # Обновляем статус задачи
                        with task_lock:
                            processing_tasks[task_id].update({
                                'progress': percent,
                                'current_frame': current_frame,
                                'processing_speed': processing_speed,
                                'status': f'Processing ({processing_speed:.1f} fps)'
                            })

                    except (ValueError, IndexError, AttributeError) as e:
                        print(f"Error parsing progress: {e}")

        # Проверяем результат выполнения
        return_code = process.poll()
        if return_code == 0:
            with task_lock:
                processing_tasks[task_id].update({
                    'completed': True,
                    'progress': 100,
                    'processed_file': os.path.basename(output_video),
                    'json_results': os.path.basename(output_json),
                    'end_time': datetime.now().strftime("%H:%M:%S"),
                    'status': 'Completed successfully'
                })
        else:
            error_msg = process.stderr.read()
            with task_lock:
                processing_tasks[task_id].update({
                    'completed': True,
                    'error': error_msg,
                    'status': f'Failed with code {return_code}'
                })

    except Exception as e:
        with task_lock:
            processing_tasks[task_id].update({
                'error': str(e),
                'status': 'Crashed',
                'is_processing': False
            })
        print(f"Error processing video {input_path}: {str(e)}")
    
    finally:
        with task_lock:
            processing_tasks[task_id]['is_processing'] = False

        # Очистка: удаляем исходный файл после обработки (опционально)
        try:
            os.remove(input_path)
        except OSError:
            pass

@app.route("/test")
def test():
    return "<h1>Test Page</h1>"

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global task_counter
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Создаем уникальное имя для выходного файла
        base_name = os.path.splitext(filename)[0]
        timestamp = int(time.time())
        output_filename = f"{base_name}_{timestamp}_processed.mp4"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        
        # Создаем папки если они не существуют
        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        file.save(input_path)
        
        with task_lock:
            task_counter += 1
            task_id = task_counter
            processing_tasks[task_id] = {
                'filename': filename,
                'upload_time': datetime.now().strftime("%H:%M:%S"),
                'is_processing': False,
                'progress': 0,
                'completed': False,
                'task_id': task_id
            }
        
        # Запускаем обработку в отдельном потоке
        thread = threading.Thread(
            target=process_video,
            args=(task_id, input_path, output_path)
        )
        thread.daemon = True  # Добавляем демонизацию потока
        thread.start()
        
        return jsonify({
            'message': 'File uploaded and processing started',
            'filename': filename,
            'task_id': task_id,
            'task_info': get_task_info()
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/progress/<int:task_id>')
def get_progress(task_id):
    with task_lock:
        task = processing_tasks.get(task_id, {})
        task_info = get_task_info()
        
        if not task:
            return jsonify({'error': 'Task not found'}), 404
        
        response = {
            'task': task,
            'task_info': task_info
        }
        
        return jsonify(response)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/tasks')
def get_tasks():
    with task_lock:
        return jsonify({
            'tasks': list(processing_tasks.values()),
            'task_info': get_task_info()
        })

if __name__ == '__main__':
    # Создаем папки, если они не существуют
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    
    # Запускаем сервер с поддержкой многопоточности
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)