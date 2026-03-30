# scenario_engine.py

import numpy as np

def match_scenario(scenario, meta, light, fft, face, prop, ai):
    """

    """
    score = 0

    # Camera (натуральная съемка)
    if scenario == "Camera":
        score += meta * 0.5           # метаданные важны
        score += light * 0.3          # физическая реалистичность
        score += face * 0.2           # лица реалистичны
        # GAN паттерн уменьшает вероятность
        score -= fft * 0.4

    # Manual Edit (ручная правка)
    elif scenario == "Manual Edit":
        score += meta * 0.3
        score += light * 0.2
        score += face * 0.3
        score += prop * 0.2           # распространение может быть естественным
        score -= fft * 0.3             # GAN артефакты снижают вероятность

    # GAN generation (синтетическое изображение)
    elif scenario == "GAN":
        score += fft * 0.6
        score += face * 0.2
        score += prop * 0.1
        score -= meta * 0.5            # отсутствие метаданных снижает вероятность
        score -= light * 0.2            # иногда освещение несовершенное

    # Deepfake (замена лица)
    elif scenario == "Deepfake":
        score += fft * 0.3
        score += (100 - face) * 0.5    # нарушения лица → характерно
        score += prop * 0.1
        score -= meta * 0.2

    # Meme / repost / compression
    elif scenario == "Meme":
        score += prop * 0.4            # распространение быстрое
        score -= fft * 0.2
        score -= face * 0.1
        score -= meta * 0.1

    else:
        # если неизвестно, явно сообщаем
        raise ValueError(f"Scenario '{scenario}' not implemented")

    return max(score, 0)  # неотрицательный score


def scenario_engine(meta, light, fft, face, prop, ai, logical, trust):

    """
    Генерация нескольких гипотез и расчет вероятностей.
    """
    # если логика плохая — подавляем "реальные" сценарии
    logic_factor = logical / 100.0
    trust_factor = trust / 100.0

    scenarios = ["Camera", "Manual Edit", "GAN", "Deepfake", "Meme"]

    # 1. Считаем score каждой гипотезы
    scores = []
    for s in scenarios:
        try:
            sc = match_scenario(s, meta, light, fft, face, prop, ai)
        except Exception as e:
            sc = 0
            print(f"Scenario '{s}' could not be evaluated:", e)
        scores.append(sc)
    scores = np.array(scores)
    # подавление нереалистичных сценариев
    for i, s in enumerate(scenarios):

        if s == "Camera":
            scores[i] *= logic_factor * trust_factor

        if s == "Manual Edit":
            scores[i] *= (0.7 * logic_factor + 0.3)

        if s == "GAN":
            scores[i] *= (1 - meta / 100)

        if s == "Deepfake":
            scores[i] *= (1 - light / 100) + (1 - logical / 100)

        if s == "Meme":
            scores[i] *= (prop / 100)

    # 2. Нормализуем в вероятности
    total = np.sum(scores)
    if total == 0:
        probs = [0 for _ in scores]  # невозможно определить вероятности
    else:
        probs = [sc / total for sc in scores]

    # 3. Победитель
    winner_idx = np.argmax(probs)
    winner = scenarios[winner_idx]

    # 4. Возврат словаря вероятностей + победитель
    prob_dict = dict(zip(scenarios, probs))
    return prob_dict, winner
def explain_scenario(winner, meta, light, fft, face, prop, ai):

    reasons = []

    if winner == "Camera":
        if meta > 50:
            reasons.append("Есть метаданные → вероятно оригинал")
        if light > 50:
            reasons.append("Освещение выглядит физически согласованным")

    if winner == "Deepfake":
        if face < 50:
            reasons.append("Обнаружены артефакты лица")
        if light < 50:
            reasons.append("Освещение не согласовано")

    if winner == "GAN":
        if fft > 50:
            reasons.append("Обнаружены частотные паттерны GAN")

    if winner == "Meme":
        if prop > 70:
            reasons.append("Характеристики повторного сжатия / распространения")

    return reasons


if __name__ == "__main__":
    # Пример использования
    meta, light, fft, face, prop, ai = 80, 70, 40, 90, 60, 75
    probs, winner = scenario_engine(meta, light, fft, face, prop, ai)
    print("Scenario probabilities:", probs)
    print("Most likely scenario:", winner)

