import os
import json
import pandas as pd
from typing import List, Dict, Tuple, Union
from transformers import AutoTokenizer

class DatasetParser:
    """
    Универсальный парсер и валидатор датасетов для Fine-Tuning.
    Поддерживает загрузку форматов: .csv, .json, .jsonl.
    """
    
    def __init__(self, tokenizer_name: str = "meta-llama/Llama-2-7b-hf"):
        # Ожидаемый стандарт: instruction, output. input и system - опционально.
        self.required_columns = {'instruction', 'output'}
        
        try:
            # Загружаем токенизатор для оценки длины датасета (необходим доступ в интернет или кэш)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            print(f"Внимание: Не удалось загрузить токенизатор {tokenizer_name}. Оценка токенов будет недоступна. Ошибка: {e}")
            self.tokenizer = None

    def load_and_validate(self, file_path: str) -> Tuple[bool, Union[List[Dict], str]]:
        """
        Загружает файл, валидирует структуру и возвращает стандартизированный массив данных.
        При ошибке возвращает (False, "Текст ошибки с указанием строки").
        """
        if not os.path.exists(file_path):
            return False, "Файл не найден."

        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == '.csv':
                data = self._parse_csv(file_path)
            elif ext == '.json':
                data = self._parse_json(file_path)
            elif ext == '.jsonl':
                data = self._parse_jsonl(file_path)
            else:
                return False, f"Неподдерживаемый формат: {ext}. Разрешены: .csv, .json, .jsonl"
            
            return self._validate_data(data)
        
        except Exception as e:
            return False, f"Критическая ошибка при чтении файла: {str(e)}"

    def _parse_csv(self, file_path: str) -> List[Dict]:
        df = pd.read_csv(file_path)
        # Заменяем NaN на пустые строки для корректной конвертации
        df = df.fillna("")
        return df.to_dict(orient='records')

    def _parse_json(self, file_path: str) -> List[Dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Корневой элемент JSON должен быть массивом объектов (list).")
            return data

    def _parse_jsonl(self, file_path: str) -> List[Dict]:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        raise ValueError(f"Синтаксическая ошибка JSONL на строке {i + 1}")
        return data

    def _validate_data(self, data: List[Dict]) -> Tuple[bool, Union[List[Dict], str]]:
        valid_data = []
        
        for i, row in enumerate(data):
            row_keys = set(row.keys())
            
            # Проверка наличия обязательных колонок
            if not self.required_columns.issubset(row_keys):
                missing = self.required_columns - row_keys
                return False, f"Ошибка в записи {i + 1}: Отсутствуют обязательные поля {missing}. Найдены ключи: {row_keys}"
            
            # Формирование стандартизированной записи, отбрасывая лишние колонки
            std_row = {
                "system": str(row.get("system", "")).strip(),
                "instruction": str(row.get("instruction", "")).strip(),
                "input": str(row.get("input", "")).strip(),
                "output": str(row.get("output", "")).strip()
            }
            
            # Базовая защита от пустых instruction/output
            if not std_row["instruction"] or not std_row["output"]:
                return False, f"Ошибка в записи {i + 1}: 'instruction' и 'output' не могут быть пустыми."

            valid_data.append(std_row)
            
        return True, valid_data

    def estimate_tokens(self, data: List[Dict]) -> int:
        """
        Подсчитывает примерное суммарное количество токенов в валидном датасете.
        Полезно для оценки стоимости в YandexGPT или времени обучения локально.
        """
        if not self.tokenizer:
            return 0
            
        total_tokens = 0
        for row in data:
            # Собираем весь текст примера для оценки
            full_text = f"{row['system']} {row['instruction']} {row['input']} {row['output']}"
            # Используем encode без возврата тензоров
            tokens = self.tokenizer.encode(full_text, add_special_tokens=True)
            total_tokens += len(tokens)
            
        return total_tokens

    def save_standardized(self, data: List[Dict], output_path: str) -> bool:
        """
        Сохраняет провалидированные данные во внутренний стандартный формат (JSONL).
        Эти файлы затем скармливаются в Local/Yandex Workers.
        """
        try:
            # Убедимся, что директория существует
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for row in data:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
            return True
        except Exception as e:
            print(f"Ошибка сохранения датасета: {e}")
            return False

# --- Пример использования модуля ---
if __name__ == "__main__":
    parser = DatasetParser()
    # is_valid, result = parser.load_and_validate("raw_data.csv")
    # if is_valid:
    #     tokens = parser.estimate_tokens(result)
    #     parser.save_standardized(result, "processed_data/dataset.jsonl")
    #     print(f"Успех. Токенов: {tokens}")
    # else:
    #     print(f"Провал валидации: {result}")