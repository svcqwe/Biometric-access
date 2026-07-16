import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from scipy.spatial.distance import cosine, euclidean
from PIL import Image

# ---------------------------------------------------------
# 1. Препроцессинг (Бинаризация -> BoundingBox -> Crop -> Resize)
# ---------------------------------------------------------
def preprocess_signature(image_input, target_size=(224, 224)) -> Image.Image:
    """
    Выполняет пайплайн предобработки подписи. 
    Принимает либо путь к файлу (str), либо PIL.Image.
    """
    if isinstance(image_input, Image.Image):
        # Конвертируем PIL Image (с экрана) в numpy массив (grayscale)
        img = np.array(image_input.convert("L"))
    elif isinstance(image_input, str):
        # Чтение изображения по пути
        img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Ошибка загрузки изображения: {image_input}")
    else:
        raise TypeError("Ожидается путь к файлу (str) или PIL Image")

    # Бинаризация (Otsu) + Инверсия
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Поиск пикселей и обрезка
    points = cv2.findNonZero(binary)
    if points is None:
        raise ValueError("На изображении не найдены пиксели подписи.")

    x, y, w, h = cv2.boundingRect(points)
    cropped = img[y:y+h, x:x+w]

    # Resize и конвертация обратно для CNN
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(resized_rgb)


# ---------------------------------------------------------
# 2. Модель: Siamese Feature Extractor (SigNet / ResNet Base)
# ---------------------------------------------------------
class SignatureExtractor(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super(SignatureExtractor, self).__init__()
        
        # Используем ResNet50 как backbone. 
        # Если у тебя есть веса специфичного SigNet, структуру нужно будет заменить.
        # В PyTorch это самый стабильный экстрактор для входа 224x224.
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Убираем последний классификационный слой (FC)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Добавляем проекционную голову для получения эмбеддингов нужной размерности
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(resnet.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        # x.shape -> [B, 3, 224, 224]
        features = self.backbone(x)
        embeddings = self.fc(features)
        
        # L2 Нормализация эмбеддингов (критически важно для косинусного расстояния)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


# ---------------------------------------------------------
# 3. Трансформации для тензоров и инференс
# ---------------------------------------------------------
def get_embedding(model: nn.Module, img: Image.Image, device: torch.device) -> np.ndarray:
    """
    Переводит изображение в тензор и прогоняет через модель.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Стандартная нормализация ImageNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device) # Добавляем batch dimension
    
    model.eval()
    with torch.no_grad():
        embedding = model(img_tensor)
        
    return embedding.cpu().numpy().flatten()


# ---------------------------------------------------------
# 4. Сравнение (Косинусное и Евклидово расстояния)
# ---------------------------------------------------------
def compare_embeddings(emb1: np.ndarray, emb2: np.ndarray):
    """
    Вычисляет расстояния между двумя эмбеддингами.
    """
    # scipy.spatial.distance.cosine возвращает (1 - cos(theta)), 
    # поэтому чем ближе к 0, тем больше похожи вектора.
    cos_dist = cosine(emb1, emb2)
    
    # Евклидово расстояние
    euc_dist = euclidean(emb1, emb2)
    
    return cos_dist, euc_dist


# ---------------------------------------------------------
# 5. Сборка всего пайплайна
# ---------------------------------------------------------
if __name__ == "__main__":
    # Настройка девайса
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Используемый device: {device}")
#
    ## Инициализация модели экстрактора
    ## Если у тебя есть предобученные веса (например, .pth после Contrastive Loss), загрузи их:
    ## model.load_state_dict(torch.load('signet_weights.pth'))
    #model = SignatureExtractor(embedding_dim=128).to(device)
    #
    #try:
    #    # Пути к изображениям подписей (замени на свои)
    #    img_path_1 = "D:/test (1).jpg"
    #    img_path_2 = "D:/test (2).jpg"
#
    #    # Шаг 1: Препроцессинг
    #    print("Препроцессинг изображений...")
    #    processed_img_1 = preprocess_signature(img_path_1)
    #    processed_img_2 = preprocess_signature(img_path_2)
#
    #    # Шаг 2: Получение эмбеддингов
    #    print("Извлечение признаков (CNN)...")
    #    emb1 = get_embedding(model, processed_img_1, device)
    #    emb2 = get_embedding(model, processed_img_2, device)
    #    print(emb2.shape)
    #    print(emb2)
#
    #    # Шаг 3: Вычисление расстояний
    #    cos_dist, euc_dist = compare_embeddings(emb1, emb2)
#
    #    print("\n=== Результаты сравнения ===")
    #    print(f"Косинусное расстояние: {cos_dist:.4f} (ближе к 0 -> подписи похожи)")
    #    print(f"Евклидово расстояние:  {euc_dist:.4f} (ближе к 0 -> подписи похожи)")
#
    #except Exception as e:
    #    print(f"Ошибка выполнения: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignatureExtractor(embedding_dim=128).to(device)
    
    # Сохраняем текущие веса (включая случайно инициализированную голову) в файл
    torch.save(model.state_dict(), 'signature_extractor_weights.pth')