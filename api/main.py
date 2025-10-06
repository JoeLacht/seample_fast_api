import logging
from contextlib import asynccontextmanager

import PIL
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import torch.nn.functional as F
from utils.model_func import (
    class_id_to_label, load_pt_model,
    load_rubert_model, transform_image
)

logger = logging.getLogger('uvicorn.info')

# Определение класса ответа для классификации изображений
class ImageResponse(BaseModel):
    class_name: str  # Название класса, например, dog, cat и т.д.
    class_index: int 

# Определение класса запроса для классификации текста
class TextInput(BaseModel):
    text: str  # Текст, введенный пользователем для классификации

# Определение класса ответа для классификации текста
class TextResponse(BaseModel):
    label: int 
    probability: float


pt_model = None  # Глобальная переменная для PyTorch модели
rb_model = None  
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Контекстный менеджер для инициализации и завершения работы FastAPI приложения.
    Загружает модели машинного обучения при запуске приложения и удаляет их после завершения.
    """
    global pt_model
    global rb_model
    global tokenizer
    # Загрузка PyTorch модели
    pt_model = load_pt_model()
    logger.info('Torch model loaded')
    # Загрузка rubert модели
    tokenizer, rb_model = load_rubert_model()
    logger.info('RB + tok model loaded')
    yield
    # Удаление моделей и освобождение ресурсов
    del pt_model, rb_model, tokenizer

app = FastAPI(lifespan=lifespan)

@app.get('/')
def return_info():
    """
    Возвращает приветственное сообщение при обращении к корневому маршруту API.
    """
    return 'Hello FastAPI!'

@app.post('/clf_image')
def classify_image(file: UploadFile):
    """
    Эндпоинт для классификации изображений.
    Принимает файл изображения, обрабатывает его, делает предсказание и возвращает название и индекс класса.
    """
    # Открытие изображения
    image = PIL.Image.open(file.file)
    # Предобработка изображения
    adapted_image = transform_image(image)
    # Логирование формы обработанного изображения
    logger.info(f'{adapted_image.shape}')
    # Предсказание класса изображения
    with torch.inference_mode():
        pred_index = pt_model(adapted_image).numpy().argmax()
    # Преобразование индекса в название класса
    imagenet_class = class_id_to_label(pred_index)
    # Формирование ответа
    response = ImageResponse(
        class_name=imagenet_class,
        class_index=pred_index
    )
    return response

@app.post('/clf_text')
def clf_text(data: TextInput):
    """
    Эндпоинт для классификации оценки отзыва на ресторан.
    """
    encoding = tokenizer(
            data.text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    # with torch.no_grad():
    #     outputs = rb_model(input_ids=input_ids, attention_mask=attention_mask)
    #     pred_class = torch.argmax(outputs, dim=1).item()
    # response = TextResponse(
    #     label=pred_class
    # )
    with torch.no_grad():
        outputs = rb_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        probs = F.softmax(logits, dim=1)  # преобразуем логиты в вероятности
        pred_class = torch.argmax(probs, dim=1).item()
        pred_prob = probs[0, pred_class].item()  # вероятность выбранного класса

    response = {
        "label": pred_class,
        "probability": pred_prob
    }
    return response

if __name__ == "__main__":
    # Запуск приложения на localhost с использованием Uvicorn
    # производится из командной строки: python your/path/api/main.py
    uvicorn.run("main:app", host='127.0.0.1', port=8000, reload=True)