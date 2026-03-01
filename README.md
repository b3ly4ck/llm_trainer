# 🚀 LLM Trainer Pro — Инструкция по деплою на сервер

## 1. Подготовка интерфейса (Фронтенд)
Папка `dist` должна лежать в корне проекта (`llm_platform/dist`). Сервер раздает интерфейс из нее. Если собирать через npm не хочется, просто создай папку и закинь туда готовый `index.html`:

```bash
mkdir -p dist
nano dist/index.html
```
*(Вставь туда код интерфейса, сохрани через `Ctrl+O`, `Enter`, `Ctrl+X`)*

---

## 2. Запуск рабочих процессов (через Screen)
Для работы платформы нужны три независимых фоновых окна.

**Окно 1: Бэкенд (FastAPI)**
Отвечает за API и раздачу интерфейса.
```bash
screen -S backend
source venv/bin/activate
uvicorn src.main:app --host 0.0.0.0 --port 8888
```
*Как свернуть: нажми **`Ctrl + A`**, затем **`D`**.*

**Окно 2: Воркер (Celery)**
Отвечает за выполнение тяжелой задачи — самого обучения.
```bash
screen -S worker
source venv/bin/activate
celery -A src.celery_app worker --loglevel=info -P solo
```
*Как свернуть: нажми **`Ctrl + A`**, затем **`D`**.*

**Окно 3: Туннель (Вывод в интернет)**
Прокидывает порт сервера наружу, чтобы открывать интерфейс с любого устройства без настроек роутера.
```bash
screen -S tunnel
ssh -R 80:localhost:8888 nokey@localhost.run
```
*Скопируй ссылку вида `https://[уникальное-имя].lhr.life` из консоли и сверни окно: **`Ctrl + A`**, затем **`D`**.*

---

## 3. Загрузка LLM модели
Модели весят десятки гигабайт. Чтобы качать их на максимальной скорости (в несколько потоков), используем `hf_transfer`.

Открой обычный терминал на сервере:
```bash
source venv/bin/activate
pip install hf_transfer huggingface_hub

# Пример многопоточной загрузки (YandexGPT-5-Lite-8B)
HF_HUB_ENABLE_HF_TRANSFER=1 python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='yandex/YandexGPT-5-Lite-8B-instruct', local_dir='/home/human/models/YandexGPT-5-Lite-8B-instruct')"
```

---

## 4. Запуск обучения в браузере
1. Открой ссылку, которую выдал туннель (`.lhr.life`).
2. В поле **Абсолютный путь к модели** укажи папку со скачанными весами (например, `/home/human/models/YandexGPT-5-Lite-8B-instruct`).
3. Загрузи датасет (формат CSV или JSONL).
4. Нажми **LoRA** (быстро и экономно) или **Full Fine-Tune** (требует много VRAM).
5. Графики Loss и логи поползут прямо в браузере.

---

## 🧯 Шпаргалка по управлению сервером
* **Посмотреть логи процесса обучения:** `screen -r worker`
* **Перезапустить зависший бэкенд:** `screen -r backend` -> нажать `Ctrl + C` -> стрелочка вверх -> `Enter`
* **Посмотреть все активные экраны:** `screen -ls`
* **Убить конкретное окно:** `screen -S ИМЯ -X quit`
* **Выжечь вообще все скрины (хард ресет):** `pkill screen`