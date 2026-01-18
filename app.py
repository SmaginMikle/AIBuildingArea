import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# ----------------------------
# 1. Настройки
# ----------------------------
SCALE_MIN = 200.0
SCALE_MAX = 3000.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Анализ спутниковых снимков", layout="wide")
st.title("Анализ спутникового снимка: площадь зданий")


# ----------------------------
# 2. Модель регрессии масштаба (SimpleScaleCNN)
# ----------------------------
class SimpleScaleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.regressor(x).squeeze(1)


@st.cache_resource
def load_scale_model():
    model = SimpleScaleCNN()
    model.load_state_dict(torch.load("Models/best_simple_cnn.pth", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


# ----------------------------
# 3. Модель сегментации (ваша UNetResNet18)
# ----------------------------
class UNetResNet18(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        encoder = resnet18(weights=weights)

        self.enc0 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.enc1 = encoder.layer1
        self.enc2 = encoder.layer2
        self.enc3 = encoder.layer3
        self.enc4 = encoder.layer4

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = self._conv_block(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = self._conv_block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = self._conv_block(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder1 = self._conv_block(128, 64)
        self.upconv0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder0 = self._conv_block(32, 32)
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e0 = self.enc0(x)
        x = self.pool(e0)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d4 = self.decoder4(torch.cat([self.upconv4(e4), e3], dim=1))
        d3 = self.decoder3(torch.cat([self.upconv3(d4), e2], dim=1))
        d2 = self.decoder2(torch.cat([self.upconv2(d3), e1], dim=1))
        d1 = self.decoder1(torch.cat([self.upconv1(d2), e0], dim=1))
        d0 = self.decoder0(self.upconv0(d1))
        return self.final(d0)


@st.cache_resource
def load_segmentation_model():
    model = UNetResNet18(num_classes=1)
    state_dict = torch.load("Models/house_segmentation_model.pth", map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model


# ----------------------------
# 4. Функции инференса
# ----------------------------
def predict_scale(model, image_pil):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_norm = model(tensor).item()
        scale = pred_norm * (SCALE_MAX - SCALE_MIN) + SCALE_MIN
    return scale


def predict_building_mask(model, image_bgr, tile_size=512, overlap=64, batch_size=8, threshold=0.5):
    """Использует вашу функцию tiled inference"""
    h, w = image_bgr.shape[:2]

    if h <= tile_size and w <= tile_size:
        pad_h = tile_size - h
        pad_w = tile_size - w
        padded = np.pad(image_bgr, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(tensor)
            prob = torch.sigmoid(output)[0, 0].cpu().numpy()
            prob = prob[:h, :w]
        return (prob > threshold).astype(np.uint8) * 255

    stride = tile_size - overlap
    prob_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    tiles = []
    coords = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = y_end - tile_size
            x_start = x_end - tile_size
            tiles.append(image_bgr[y_start:y_end, x_start:x_end])
            coords.append((y_start, y_end, x_start, x_end))

    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i + batch_size]
            batch_coords = coords[i:i + batch_size]

            tensors = []
            for tile in batch_tiles:
                pad_h = tile_size - tile.shape[0]
                pad_w = tile_size - tile.shape[1]
                if pad_h > 0 or pad_w > 0:
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
                tensor = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0
                tensors.append(tensor)

            batch_tensor = torch.stack(tensors).to(DEVICE)
            outputs = model(batch_tensor)
            probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()

            for j, (y1, y2, x1, x2) in enumerate(batch_coords):
                prob_tile = probs[j]
                prob_tile = prob_tile[:y2 - y1, :x2 - x1]
                prob_sum[y1:y2, x1:x2] += prob_tile
                weight_sum[y1:y2, x1:x2] += 1

    avg_prob = np.divide(prob_sum, weight_sum, where=weight_sum > 0)
    return (avg_prob > threshold).astype(np.uint8) * 255


# ----------------------------
# 5. UI Streamlit
# ----------------------------
uploaded_file = st.file_uploader("Загрузите спутниковый снимок (.jpg, .png, .tif)",
                                 type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.getvalue())
        img_path = tmp.name

    try:
        # Загружаем как PIL (для масштаба) и OpenCV (для сегментации)
        image_pil = Image.open(img_path).convert("RGB")
        image_bgr = cv2.imread(img_path)

        st.subheader("Исходное изображение")
        st.image(image_pil, caption="Загруженный снимок")

        # Загрузка моделей
        scale_model = load_scale_model()
        seg_model = load_segmentation_model()

        # Прогноз масштаба
        with st.spinner("Определение масштаба..."):
            scale = predict_scale(scale_model, image_pil)

        # Прогноз сегментации
        with st.spinner("Выделение зданий (может занять время для больших изображений)..."):
            building_mask = predict_building_mask(seg_model, image_bgr)

        # Доля зданий
        total_pixels = building_mask.size
        building_pixels = np.sum(building_mask > 0)
        building_ratio = building_pixels / total_pixels

        # Вывод результатов
        st.subheader("Результаты анализа")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Масштаб (длина стороны)", f"{scale:.1f} м")
        with col2:
            st.metric("Доля площади под зданиями", f"{building_ratio:.2%}")
        with col3:
            st.metric("Площадь зданий", f"{scale * scale * building_ratio:.1f} м^2:")

        st.subheader("Маска зданий")
        st.image(building_mask, caption="Белое — здания, чёрное — всё остальное", clamp=True)

    except Exception as e:
        st.error(f"Ошибка при обработке: {e}")
    finally:
        os.unlink(img_path)
else:
    st.info("Пожалуйста, загрузите изображение для анализа.")