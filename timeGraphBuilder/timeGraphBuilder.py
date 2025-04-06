import matplotlib.pyplot as plt
import re

# Функция для парсинга строк и извлечения списков значений
def parse_time_results(file_path):
    results = {}
    with open(file_path, "r") as file:
        for line in file:
            match = re.search(r'^(.*?): Avg: .*? Max: .*? \| (\[.*\])$', line.strip())
            if match:
                key = match.group(1).strip()  # Название операции
                values = eval(match.group(2))  # Преобразуем строку в список чисел
                results[key] = [v * 1000 for v in values]
    return results

# Функция для построения единого графика
def plot_results(results, save=False):
    plt.figure(figsize=(12, 6))

    for key, values in results.items():
        plt.plot(range(1, len(values) + 1), values, marker='o', linestyle='-', label=key)

    plt.xlabel("Measurement Number")
    plt.ylabel("Time (ms)")
    plt.title("Performance Over Time")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))    # Добавляем легенду
    plt.grid(True)

    if save:
        plt.savefig("performance_over_time.png")
    else:
        plt.show()

# Основной код
if __name__ == "__main__":
    file_path = "timeResults.txt"  # Файл с данными
    results = parse_time_results(file_path)  # Парсим данные

    plot_results(results, save=True)  # Построение графика (save=True — сохранить)
