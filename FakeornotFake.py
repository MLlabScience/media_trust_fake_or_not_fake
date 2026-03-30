#FakeornotFake.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import exifread
import matplotlib.pyplot as plt


st.set_page_config(page_title="MediaTrust AI", layout="wide")

st.title("MediaTrust AI")
st.subheader("Hybrid Deepfake Detection System")

st.write("Загрузите изображение для анализа подлинности с использованием метаданных, физических принципов, GAN-отпечатков и анализа с помощью ИИ.")

uploaded_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])

############################################
# METADATA ANALYSIS
############################################

def metadata_analysis(file):
    score = 100
    warnings = []

    try:
        file.seek(0)
        tags = exifread.process_file(file)

        if not tags:
            score -= 40
            warnings.append("Метаданные не найдены")

        if "Image Software" in tags:
            software = str(tags["Image Software"])
            if "Photoshop" in software or "GIMP" in software:
                score -= 30
                warnings.append("Обнаружено программное обеспечение для редактирования")

        if "EXIF DateTimeOriginal" not in tags:
            score -= 20
            warnings.append("Отсутствует исходная метка времени.")

    except:
        score -= 50
        warnings.append("Ошибка синтаксического анализа метаданных")

    return max(score, 0), warnings

############################################
# LIGHT ANALYSIS
############################################

def light_analysis(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(gray, cv2.CV_64F,1,0)
    grad_y = cv2.Sobel(gray, cv2.CV_64F,0,1)

    angles = np.arctan2(grad_y, grad_x)

    # разброс направлений света
    direction_consistency = np.std(angles)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    intensity_consistency = np.std(magnitude)

    score = 100 - (direction_consistency * 20 + intensity_consistency)

    return np.clip(score, 0, 100)


############################################
# FFT GAN DETECTOR
############################################

def fft_analysis(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    magnitude_spectrum = np.log(np.abs(fshift)+1)

    score = np.std(magnitude_spectrum)

    score = min(score * 2, 100)

    return score

############################################
# FACE ARTIFACT DETECTION
############################################

def face_analysis(image):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    if len(faces) == 0:
        return 50  # нет лица — нейтрально

    scores = []

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]

        blur = cv2.Laplacian(face, cv2.CV_64F).var()
        symmetry = np.mean(np.abs(face - np.fliplr(face)))
        f = np.fft.fft2(face)
        fshift = np.fft.fftshift(f)
        spectrum = np.log(np.abs(fshift) + 1)

        freq_noise = np.std(spectrum)

        score = 100 - (blur * 0.2 + symmetry * 0.2 + freq_noise * 0.3)
        scores.append(score)

    return max(0, np.mean(scores))

############################################
# HEATMAP
############################################

def generate_heatmap(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(9,9),0)
    diff = cv2.absdiff(gray,blur)
    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    return heatmap

############################################
# PROPAGATION (заглушка заменена логикой)
############################################
def get_format(pil_image):
    if hasattr(pil_image, "format") and pil_image.format:
        return pil_image.format.lower()
    return "unknown"


def propagation_analysis(image_cv, pil_image):

    fmt = get_format(pil_image)

    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    score = 100

    #################################
    # 1. JPEG recompression test
    #################################
    if fmt in ["jpeg", "jpg", "webp"]:

        _, enc = cv2.imencode('.jpg', gray, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        dec = cv2.imdecode(enc, 0)

        diff = np.mean(np.abs(gray - dec))

        score -= diff * 0.6

    if fmt == "jpeg" and not pil_image.info:
        score -= 15
    _, enc90 = cv2.imencode('.jpg', gray, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    dec90 = cv2.imdecode(enc90, 0)

    double_diff = 0

    if fmt in ["jpeg", "jpg", "webp"]:
        _, enc = cv2.imencode('.jpg', gray, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        dec = cv2.imdecode(enc, 0)

        _, enc90 = cv2.imencode('.jpg', gray, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        dec90 = cv2.imdecode(enc90, 0)

        double_diff = np.mean(np.abs(dec - dec90))

        if double_diff < 2:
            score -= 10

    b, g, r = cv2.split(image_cv)

    channel_diff = np.std(b - g) + np.std(g - r)

    score -= channel_diff * 0.05

    #################################
    # 2. Noise consistency
    #################################
    noise = np.std(gray)
    score -= noise * 0.2

    #################################
    # 3. Blocking artifacts (очень важно)
    #################################
    h, w = gray.shape

    block_diff = 0
    for i in range(8, h, 8):
        block_diff += np.mean(np.abs(gray[i] - gray[i-1]))

    for j in range(8, w, 8):
        block_diff += np.mean(np.abs(gray[:, j] - gray[:, j-1]))

    score -= block_diff * 0.05

    #################################
    # 4. PNG / lossless проверка
    #################################
    if fmt in ["png", "tiff"]:

        # PNG не должен иметь JPEG-артефактов
        if noise > 25:
            score -= 20  # странно для lossless

        if block_diff > 10:
            score -= 25  # подозрение на перекодировку

    #################################

    return np.clip(score, 0, 100)


############################################
# AI DETECTION (простая модель)
############################################

def ai_detection_score(fft_score, face_score):
    return (fft_score * 0.6 + (100 - face_score) * 0.4)

############################################
# QUANTUM-INSPIRED TRUST
############################################

############################################
def feature_correlation(meta, light, fft, face, prop, ai):

    features = np.array([meta, light, fft, face, prop, ai])

    # нормализация
    features = features / 100.0

    matrix = np.outer(features, features)

    return matrix
############################################
def quantum_features(meta, light, fft, face, prop, ai):

    vec = np.array([meta, light, fft, face, prop, ai]) / 100.0

    # "запутанность"
    entanglement = vec[0]*vec[1] + vec[2]*vec[3]

    # "интерференция"
    interference = np.dot(vec, vec[::-1])

    # "дисбаланс"
    variance = np.var(vec)

    return entanglement, interference, variance

def classify_case(meta, light, fft, face, variance, conflicts):

    if not conflicts and variance < 0.03 and face < 30:
        return "AI-generated image likely"

    if conflicts and light < 40:
        return "Deepfake likely"

    return "Uncertain"

import pandas as pd

def build_feature_table(meta, light, fft, face, prop, ai):

    data = [
        ["Metadata", meta, "Наличие и целостность EXIF-данных", "Низкое значение → подозрительный источник"],
        ["Lighting", light, "Физическая согласованность освещения", "Низкое значение → возможный дипфейк"],
        ["FFT (GAN)", fft, "Частотные паттерны генеративных моделей", "Высокое значение → GAN-артефакты"],
        ["Face", face, "Анализ лица (симметрия, текстура)", "Низкое значение → артефакты генерации"],
        ["Propagation", prop, "Поведение распространения", "Низкое → подозрительное происхождение"],
        ["AI Score", ai, "Комплексная оценка ИИ", "Низкое → высокая вероятность фейка"]
    ]

    df = pd.DataFrame(data, columns=[
        "Признак", "Значение", "Описание", "Интерпретация"
    ])

    return df

def probability_estimation(meta, light, fft, face, ai, prop, logical_score, nn):
    deepfake_score = (
            (100 - light) * 0.2 +
            (100 - face) * 0.2 +
            (100 - ai) * 0.15 +
            (100 - prop) * 0.1 +
            (100 - logical_score) * 0.1 +
            nn * 0.25  # ←
    )

    fft_weight = 0.15

    ai_score_prob = (
            fft * 0.15 +
            (100 - face) * 0.2 +
            prop * 0.15 +
            logical_score * 0.1 +
            nn * 0.4  # ←
    )

    real_score = (
            meta * 0.25 +
            light * 0.25 +
            face * 0.2 +
            prop * 0.15 +
            (100 - nn) * 0.15  # ←
    )

    total = deepfake_score + ai_score_prob + real_score

    if total == 0:
        return 33, 33, 33

    return (
        real_score / total * 100,
        deepfake_score / total * 100,
        ai_score_prob / total * 100
    )

from transformers import ViTImageProcessor, AutoModelForImageClassification
import torch

@st.cache_resource
def load_model():
    processor = ViTImageProcessor.from_pretrained("prithivMLmods/Deepfake-Detection-Exp-02-21")
    model = AutoModelForImageClassification.from_pretrained("prithivMLmods/Deepfake-Detection-Exp-02-21")
    return processor, model

def neural_detection(image):
    processor, model = load_model()

    image = image.convert("RGB")  #

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    score = probs[0][1].item() * 100

    return score


def risk_level(trust_score):

    if trust_score > 70:
        return "Низкий риск", "Контент выглядит достоверным"

    elif trust_score > 40:
        return "Средний риск", "Есть признаки манипуляции"

    else:
        return "Высокий риск", "Высокая вероятность дипфейка или синтетики"

def plot_risk_dynamics(meta, light, fft, face, ai):

    labels = ["Metadata↓", "Lighting↓", "FFT↑", "Face↓", "AI↓"]

    values = [
        100 - meta,
        100 - light,
        fft,
        100 - face,
        100 - ai
    ]

    fig, ax = plt.subplots()
    ax.plot(labels, values, marker='o')

    ax.set_title("Факторы, увеличивающие вероятность фейка")
    ax.set_ylabel("Влияние")

    return fig


def dynamic_explanation(meta, light, fft, face, ai):

    explanations = []

    if meta < 50:
        explanations.append("Метаданные отсутствуют или повреждены → источник недостоверен")

    if light < 50:
        explanations.append("Обнаружена несогласованность освещения → возможная генерация")

    if fft > 50:
        explanations.append("Выявлены частотные GAN-паттерны")

    if face < 50:
        explanations.append("Обнаружены артефакты лица → возможный deepfake")

    if ai < 40:
        explanations.append("Низкая уверенность модели → общий риск повышен")

    return explanations

def quantum_logic_score(vec):

    # ожидаемые зависимости (как "физика мира")
    expected = np.array([
        vec[0]*vec[1],        # metadata ↔ lighting
        vec[1]*vec[3],        # lighting ↔ face
        vec[2]*(1 - vec[3]),  # GAN ↔ anti-face
        vec[4]*vec[0],        # propagation ↔ metadata
        vec[5]*(1 - vec[0])   # AI ↔ отсутствие метаданных
    ])

    # реальные зависимости
    actual = np.array([
        vec[0]*vec[1],
        vec[1]*vec[3],
        vec[2]*vec[3],
        vec[4]*vec[0],
        vec[5]*vec[0]
    ])

    mismatch = np.abs(expected - actual)
    if vec[1] < 0.3 and vec[3] > 0.8:
        mismatch += 0.3

    return np.mean(mismatch)

def quantum_trust(meta, light, fft, face, prop, ai, logic):

    vec = np.array([meta, light, fft, face, prop, ai, logic]) / 100.0

    weights = np.array([0.18, 0.18, 0.18, 0.14, 0.12, 0.10, 0.10])

    base_score = np.sum(vec * weights)

    interaction = (
        vec[0]*vec[1] +
        vec[2]*vec[3] +
        vec[4]*vec[5] +
        vec[6]*vec[3]   # логика ↔ лицо
    )

    score = (0.7 * base_score + 0.3 * interaction) * 100

    return np.clip(score, 0, 100)


def explain_logic(meta, light, fft, face, prop, ai, logical):

    text = []

    # Общий уровень
    if logical < 40:
        text.append("❗ Обнаружены серьёзные логические противоречия между признаками.")
    elif logical < 70:
        text.append("⚠️ Частичная несогласованность признаков.")
    else:
        text.append("✅ Признаки логически согласованы.")

    # Детали
    if light < 40 and face > 70:
        text.append("Лицо выглядит реалистично, но освещение противоречит этому → возможный дипфейк.")

    if fft > 50 and face > 70:
        text.append("Обнаружены GAN-паттерны при высокой реалистичности лица → синтетика высокого качества.")

    if meta < 40 and face > 70:
        text.append("Реалистичное лицо без метаданных → подозрительное происхождение.")
    if meta < 40 and prop > 70:
        text.append("Контент активно распространяется без источника → возможная информационная атака.")

    if np.var([meta, light, fft, face]) < 0.05:
        text.append("Слишком идеальная согласованность → вероятна генерация нейросетью.")

    if ai < 30:
        text.append("Общая уверенность системы низкая → результат требует осторожной интерпретации.")

    return text

def quantum_decision(vec):

    # 3 состояния
    real_state = np.array([0.8, 0.8, 0.2, 0.8, 0.7, 0.3])
    ai_state = np.array([0.2, 0.7, 0.8, 0.3, 0.3, 0.9])
    fake_state = np.array([0.3, 0.3, 0.7, 0.7, 0.4, 0.8])

    # косинусная близость
    def similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)

    real_sim = similarity(vec, real_state)
    ai_sim = similarity(vec, ai_state)
    fake_sim = similarity(vec, fake_state)

    return real_sim, ai_sim, fake_sim


def classify_content(meta, light, fft, face, ai, logical):

    # 1. AI генерация
    if logical > 70 and face < 40 and fft > 40:
        return "AI-generated", "Низкий риск (синтетическое изображение)"

    # 2. Deepfake (опасно)
    if logical < 45 and face > 60 and light < 50:
        return "Deepfake", "Высокий риск (манипуляция реальностью)"

    # 3. Подозрительное
    if logical < 60:
        return "Suspicious", "Средний риск"

    # 4. Реальное
    return "Real", "Низкий риск"

def final_explanation(label, logical, real_p, deepfake_p, ai_p):

    text = []

    text.append(f"Тип контента: {label}")

    if label == "AI-generated":
        text.append("Изображение демонстрирует признаки синтетической генерации.")
        text.append("При этом логическая согласованность сохраняется → угрозы не выявлено.")

    if label == "Deepfake":
        text.append("Обнаружены противоречия между признаками.")
        text.append("Имеются признаки манипуляции реальным контентом.")

    if logical < 50:
        text.append("Ключевой фактор: нарушение логической согласованности признаков.")

    text.append(f"Вероятности → Real: {round(real_p,1)}%, Fake: {round(deepfake_p,1)}%, AI: {round(ai_p,1)}%")

    return text



# MAIN PIPELINE
############################################

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Uploaded image", width=400)

    st.write("Выполнение анализа...")

    # анализ
    meta_score, warnings = metadata_analysis(uploaded_file)
    light_score = light_analysis(image_cv)
    fft_score = fft_analysis(image_cv)
    face_score = face_analysis(image_cv)
    prop_score = propagation_analysis(image_cv, image)
    nn_score = neural_detection(image)
    ai_score_classic = ai_detection_score(fft_score, face_score)

    #
    ai_score = (
            ai_score_classic * 0.4 +
            nn_score * 0.6
    )

    # ===============================
    # КВАНТОВАЯ ЛОГИКА (НОВЫЙ БЛОК)
    # ===============================

    vec = np.array([
        meta_score,
        light_score,
        fft_score,
        face_score,
        prop_score,
        ai_score
    ]) / 100.0

    logic_mismatch = quantum_logic_score(vec)
    logical_score = np.clip(100 - logic_mismatch * 100, 0, 100)

    # ❗ штраф за "согласованно плохие признаки"
    if np.mean([meta_score, light_score, face_score]) < 40:
        logical_score *= 0.5

    label, risk_text = classify_content(
        meta_score,
        light_score,
        fft_score,
        face_score,
        ai_score,
        logical_score
    )

    trust_score = quantum_trust(
        meta_score,
        light_score,
        fft_score,
        face_score,
        prop_score,
        ai_score,
        logical_score
    )
    st.subheader("Тип контента")

    st.write(f"Тип: {label}")
    st.write(f"Риск: {risk_text}")

    st.metric("Trust Score", round(trust_score, 2))

    ############################################
    st.subheader("Матрица взаимодействия функций")

    corr_matrix = feature_correlation(
        meta_score, light_score, fft_score,
        face_score, prop_score, ai_score
    )

    fig, ax = plt.subplots()
    ax.imshow(corr_matrix)

    labels = ['Meta', 'Light', 'FFT', 'Face', 'Prop', 'AI']
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    st.pyplot(fig)
    ############################################
    st.subheader("Динамика факторов риска")

    risk_fig = plot_risk_dynamics(
        meta_score, light_score, fft_score,
        face_score, ai_score
    )

    st.pyplot(risk_fig)
    ############################################
    st.subheader("Аналитика признаков")

    df = build_feature_table(
        meta_score, light_score, fft_score,
        face_score, prop_score, ai_score
    )

    st.dataframe(df)
    ############################################
    st.subheader("Вероятностная оценка")

    real_p, deepfake_p, ai_p = probability_estimation(
        meta_score,
        light_score,
        fft_score,
        face_score,
        ai_score,
        prop_score,
        logical_score,
        nn_score
    )

    st.write(f"Реальное изображение: {round(real_p, 2)}%")
    st.write(f"Deepfake: {round(deepfake_p, 2)}%")
    st.write(f"AI-сгенерированное: {round(ai_p, 2)}%")
    ############################################
    level, desc = risk_level(trust_score)

    st.subheader("Оценка риска")

    st.error(f"{level}: {desc}")

    real_sim, ai_sim, fake_sim = quantum_decision(vec)
    ############################################
    st.subheader("Квантовая классификация")

    st.write(f"Real similarity: {round(real_sim, 2)}")
    st.write(f"AI similarity: {round(ai_sim, 2)}")
    st.write(f"Fake similarity: {round(fake_sim, 2)}")

    ############################################
    st.markdown("### Интерпретация для данного изображения")

    explanations = dynamic_explanation(
        meta_score, light_score, fft_score,
        face_score, ai_score
    )

    for e in explanations:
        st.write("•", e)


    ############################################

    def detect_conflicts(meta, light, fft, face, prop, ai):

        conflicts = []

        if meta < 50 and prop > 70:
            conflicts.append("Высокая скорость распространения, но слабые метаданные → возможна вирусная подделка")

        if light < 40 and face > 70:
            conflicts.append("Хорошее лицо, но присутсвует непостоянное освещение → возможно, это дипфейк")

        if fft > 60 and face < 50:
            conflicts.append("Обнаружен паттерн GAN + слабая структура лица")

        if ai < 30:
            conflicts.append("В целом, уровень доверия к ИИ низок.")

        return conflicts

    ############################################
    st.subheader("Обнаружены несоответствия")

    conflicts = detect_conflicts(
        meta_score, light_score, fft_score,
        face_score, prop_score, ai_score
    )

    if conflicts:
        for c in conflicts:
            st.error(c)
    else:
        st.success("Серьёзных несоответствий не обнаружено.")
    ############################################
    st.subheader("Квантовое пространство характеристик")

    ent, inter, var = quantum_features(
        meta_score, light_score, fft_score,
        face_score, prop_score, ai_score
    )

    st.write(f"Запутывание: {round(ent, 3)}")
    st.write(f"Интерференция: {round(inter, 3)}")
    st.write(f"Дисперсия: {round(var, 3)}")
    ############################################
    st.subheader("Результаты анализа")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Metadata", round(meta_score,2))
        st.metric("Lighting", round(light_score,2))

    with col2:
        st.metric("FFT (GAN)", round(fft_score,2))
        st.metric("Face Analysis", round(face_score,2))

    with col3:
        st.metric("AI Score", round(ai_score,2))
        st.metric("Trust Score", round(trust_score,2))
        st.metric("Logical Consistency", round(logical_score, 2))

    ############################################

    if warnings:
        st.warning("Metadata Issues:")
        for w in warnings:
            st.write("-", w)

    ############################################

    st.subheader("Тепловая карта аномалий")

    heatmap = generate_heatmap(image_cv)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    st.image(heatmap_rgb, caption="Suspicious regions")

    ############################################
    def plot_radar(meta, light, fft, face, prop, ai):

        labels = ['Metadata', 'Light', 'FFT', 'Face', 'Propagation', 'AI']
        values = [meta, light, fft, face, prop, ai]

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))

        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)

        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.3)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)

        return fig


    def plot_contributions(meta, light, fft, face, prop, ai):

        labels = ['Metadata', 'Light', 'FFT', 'Face', 'Propagation', 'AI']
        values = [meta, light, fft, face, prop, ai]

        fig, ax = plt.subplots()
        ax.bar(labels, values)

        ax.set_ylabel("Score")
        ax.set_title("Вклад в функции")

        return fig


    st.subheader("Системная интерпретация")

    radar_fig = plot_radar(
        meta_score, light_score, fft_score,
        face_score, prop_score, ai_score
    )

    st.pyplot(radar_fig)

    bar_fig = plot_contributions(
        meta_score, light_score, fft_score,
        face_score, prop_score, ai_score
    )

    st.pyplot(bar_fig)

    st.subheader("Логическая интерпретация")

    logic_text = explain_logic(
        meta_score,
        light_score,
        fft_score,
        face_score,
        prop_score,
        ai_score,
        logical_score
    )

    for t in logic_text:
        st.write("•", t)
    ############################################
    from scenario_engine import scenario_engine, explain_scenario

    scenario_probs, scenario_winner = scenario_engine(
        meta_score,
        light_score,
        fft_score,
        face_score,
        prop_score,
        ai_score,
        logical_score,
        trust_score
    )

    st.write("Scenario probabilities:", scenario_probs)
    st.write("Most likely scenario:", scenario_winner)

    ############################################
    st.subheader("Сценарии происхождения")

    import matplotlib.pyplot as plt

    labels = list(scenario_probs.keys())
    values = [v * 100 for v in scenario_probs.values()]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("Probability %")
    ax.set_title("Scenario likelihood")

    st.pyplot(fig)
    st.success(f"Наиболее вероятный сценарий: {scenario_winner}")
    st.markdown("### Интерпретация сценария")

    if scenario_winner == "Camera":
        st.success("Изображение похоже на оригинальную съёмку")

    elif scenario_winner == "Deepfake":
        st.error("Высокая вероятность подмены лица (deepfake)")

    elif scenario_winner == "GAN":
        st.warning("Изображение может быть сгенерировано нейросетью")

    elif scenario_winner == "Manual Edit":
        st.info("Вероятно редактирование вручную")

    elif scenario_winner == "Meme":
        st.info("Похоже на репост / сжатие / мем")

    ############################################
    st.subheader("Почему система так решила")

    reasons = explain_scenario(
        scenario_winner,
        meta_score, light_score, fft_score,
        face_score, prop_score, ai_score
    )

    for r in reasons:
        st.write("•", r)

    ############################################
    if trust_score > 75:
        st.success("Достоверный")

    elif trust_score > 50:
        st.warning("Средний риск")

    else:
        st.error("Дипфейк скорее всего")
