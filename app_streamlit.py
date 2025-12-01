# -*- coding: utf-8 -*-
import os
import json
import tempfile
from pathlib import Path

import numpy as np
import cv2
import streamlit as st

from app.pose_extractor import extract_pose_from_video, get_video_frames
from app.preprocessing import normalize_landmarks, compute_angles_sequence, smooth_series
from app.dtw_utils import stack_features, align_by_dtw
from app.visualization import draw_skeleton, make_side_by_side

# ---------------------------- Пути и конфиг ----------------------------

CONFIG_PATH = Path("app/elements_config.json")
REF_DIR = Path("references")
REF_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="AI-коуч: покадровое сравнение", layout="wide")
st.title("🤸 AI-коуч: покадровое сравнение техники")

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(cfg: dict):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

# ---------------------------- Вспомогательные функции ----------------------------

def _md5_bytes(b: bytes) -> str:
    import hashlib
    h = hashlib.md5()
    h.update(b)
    return h.hexdigest()

if "user_video_path" not in st.session_state:
    st.session_state.user_video_path = None
if "user_video_hash" not in st.session_state:
    st.session_state.user_video_hash = None
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "selected_element" not in st.session_state:
    st.session_state.selected_element = None

def _save_upload_to_tmp(uploaded_file):
    data = uploaded_file.read()
    h = _md5_bytes(data)
    tmp_dir = Path(tempfile.gettempdir()) / "ai_coach_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / f"{h}.mp4"
    if not out_path.exists():
        with open(out_path, "wb") as f:
            f.write(data)
    st.session_state.user_video_path = str(out_path)
    st.session_state.user_video_hash = h
    return st.session_state.user_video_path, h

@st.cache_data(show_spinner=False)
def run_analysis_cached(user_video_hash, user_video_path, cfg_json, ref_mtime):
    """Пайплайн анализа: поза → нормализация → углы → DTW."""
    cfg = json.loads(cfg_json)
    ref_path = cfg["reference_video"]

    # 1. Позы
    user_seq = extract_pose_from_video(user_video_path)
    ref_seq = extract_pose_from_video(ref_path)

    # 2. Нормализация и углы
    user_norm = normalize_landmarks(user_seq.landmarks)
    ref_norm = normalize_landmarks(ref_seq.landmarks)
    user_angles = smooth_series(compute_angles_sequence(user_norm), 11, 3)
    ref_angles = smooth_series(compute_angles_sequence(ref_norm), 11, 3)

    # 3. DTW по набору признаков
    feat_keys = [
        k for k in user_angles.keys()
        if any(s in k for s in ["torso", "hip", "knee"])
    ] or list(user_angles.keys())[:6]

    user_feats = stack_features(user_angles, feat_keys)
    ref_feats = stack_features(ref_angles, feat_keys)

    _, _, path = align_by_dtw(user_feats, ref_feats)
    idx_user = [p[0] for p in path]
    idx_ref = [p[1] for p in path]

    # 4. Ошибки по углам (усреднённо по выровненным траекториям)
    angle_mae = {}
    for k in set(user_angles.keys()).intersection(ref_angles.keys()):
        angle_mae[k] = float(
            np.nanmean(np.abs(user_angles[k][idx_user] - ref_angles[k][idx_ref]))
        )

    return {
        "user_fps": user_seq.fps,
        "ref_fps": ref_seq.fps,
        "user_landmarks_raw": user_seq.landmarks,
        "ref_landmarks_raw": ref_seq.landmarks,
        "user_angles": user_angles,
        "ref_angles": ref_angles,
        "idx_user": idx_user,
        "idx_ref": idx_ref,
        "path": path,
        "angle_mae": angle_mae,
    }

@st.cache_data(show_spinner=False)
def load_frames_cached(video_path: str):
    return get_video_frames(video_path)

# --------- генерация советов с учётом особенностей элемента ---------

def generate_element_tips(angle_mae: dict, element_cfg: dict) -> list[str]:
    """
    Строит советы по улучшению:
    - учитывает среднюю ошибку по группам суставов;
    - учитывает важность суставов из element_cfg["important_joints"];
    - использует пороги из element_cfg["tips_thresholds_deg"].
    """
    if not angle_mae:
        return ["Недостаточно данных для анализа техники. Попробуйте записать более чёткое видео."]

    title = element_cfg.get("title", "элемент")
    thresholds = element_cfg.get("tips_thresholds_deg", {"minor": 8, "major": 18})
    t_minor = thresholds.get("minor", 8.0)
    t_major = thresholds.get("major", 18.0)
    important = element_cfg.get("important_joints", {})

    # группы суставов: (id, ключ для important_joints, список углов, формы слова)
    groups = [
        ("плечи",   "shoulder", ["shoulder_left", "shoulder_right"], ("плечи", "плеч", "плечи")),
        ("локти",   "elbow",    ["elbow_left", "elbow_right"],       ("локти", "локтей", "локти")),
        ("корпус",  "torso",    ["torso"],                           ("корпус", "корпуса", "корпус")),
        ("бедра",   "hip",      ["hip_left", "hip_right"],           ("бедра", "бедра", "бедро/таз")),
        ("колени",  "knee",     ["knee_left", "knee_right"],         ("колени", "коленей", "колени")),
        ("лодыжки", "ankle",    ["ankle_left", "ankle_right"],       ("лодыжки", "лодыжек", "лодыжки")),
    ]

    stats = []  # (score, err, group_id, forms, joint_key)
    for group_id, imp_key, keys, forms in groups:
        vals = [angle_mae[k] for k in keys if k in angle_mae and np.isfinite(angle_mae[k])]
        if not vals:
            continue
        err = float(np.mean(vals))
        weight = float(important.get(imp_key, 1.0))
        score = err * weight
        stats.append((score, err, group_id, forms, imp_key))

    if not stats:
        return ["Техника близка к эталону: заметных отклонений по суставам не обнаружено."]

    # сортируем по "важность × ошибка", чтобы учитывать особенности элемента
    stats.sort(reverse=True, key=lambda x: x[0])

    tips = []
    for score, err, group_id, forms, imp_key in stats:
        if err < t_minor:
            continue  # совсем мелкие отклонения пропускаем
        nom, rod, vin = forms
        weight = float(important.get(imp_key, 1.0))

        if err >= t_major:
            txt = (
                f"Для элемента «{title}» критично положение **{rod}**. "
                f"Сейчас средняя ошибка по этой зоне ≈ **{err:.1f}°**. "
            )
            if weight > 1.5:
                txt += "Этот участок отмечен как ключевой для данного элемента, "
            txt += (
                "потренируйте движение с акцентом на стабильность: выполняйте элемент медленнее, "
                f"контролируя {vin} в зеркале или по видео, добивайтесь одинакового положения в начале и в конце фазы."
            )
        else:
            txt = (
                f"Обратите внимание на **{vin}** — средняя ошибка ≈ **{err:.1f}°**. "
                f"Для элемента «{title}» важно, чтобы {nom} не «гуляли». "
                "Попробуйте уменьшить амплитуду и сосредоточиться на точном воспроизведении угла."
            )
        tips.append(txt)

    if not tips:
        tips.append("Выполнение элемента в пределах допустимой погрешности. Можно усложнять упражнение или увеличивать амплитуду.")
    # ограничим количество советов, чтобы не было «простыней»
    return tips[:4]

# ---------------------------- Вкладки ----------------------------

tab_analyze, tab_editor = st.tabs(["Анализ видео", "Редактор элементов"])

# ======================================================================
#                           ВКЛАДКА: АНАЛИЗ
# ======================================================================
with tab_analyze:
    cfg = load_config()
    if not cfg:
        st.warning("Конфигурация элементов пуста. Сначала добавьте элемент во вкладке «Редактор элементов».")
    else:
        elements = list(cfg.keys())
        default_el = st.session_state.selected_element or (elements[0] if elements else None)
        el = st.selectbox(
            "Выберите элемент",
            elements,
            index=elements.index(default_el) if default_el in elements else 0,
            format_func=lambda k: cfg[k].get("title", k),
        )
        st.session_state.selected_element = el

        st.caption("Для сравнения используется эталонное видео, указанное в настройках элемента.")

        user_file = st.file_uploader(
            "Видео пользователя (mp4/avi/mov/mkv)",
            type=["mp4", "avi", "mov", "mkv"],
            key="user_video_upload",
        )
        analyze_clicked = st.button("Анализировать", type="primary", key="analyze_btn")

        if analyze_clicked:
            if not user_file:
                st.error("Пожалуйста, загрузите видео пользователя.")
                st.stop()
            user_path, user_hash = _save_upload_to_tmp(user_file)
            ref_path = cfg[el]["reference_video"]
            if not os.path.exists(ref_path):
                st.error(f"Не найден файл эталона: {ref_path}")
                st.stop()
            ref_mtime = os.path.getmtime(ref_path)
            with st.spinner("Выполняется анализ и выравнивание..."):
                st.session_state.analysis = run_analysis_cached(
                    user_hash,
                    user_path,
                    json.dumps(cfg[el], ensure_ascii=False),
                    ref_mtime,
                )

        if st.session_state.analysis is None:
            st.info("Загрузите видео и нажмите «Анализировать», чтобы увидеть покадровое сравнение и советы.")
        else:
            A = st.session_state.analysis

            # Загружаем кадры
            _, user_frames = load_frames_cached(st.session_state.user_video_path)
            _, ref_frames = load_frames_cached(cfg[el]["reference_video"])

            # Считаем ошибку по каждому выровненному кадру
            aligned_len = len(A["idx_user"])
            per_frame_err = np.zeros(aligned_len, float)
            angle_keys = list(A["angle_mae"].keys())
            for i in range(aligned_len):
                s = 0.0
                c = 0
                for k in angle_keys:
                    u = A["user_angles"][k][A["idx_user"]][i]
                    r = A["ref_angles"][k][A["idx_ref"]][i]
                    if not (np.isnan(u) or np.isnan(r)):
                        s += abs(u - r)
                        c += 1
                per_frame_err[i] = (s / c) if c else np.nan
            per_frame_err = np.nan_to_num(per_frame_err, nan=0.0)

            # ---- Советы с учётом элемента ----
            st.markdown("### Советы по улучшению выполнения")
            tips = generate_element_tips(A["angle_mae"], cfg[el])
            for t in tips:
                st.write("• " + t)

            # ---- Покадровое сравнение ----
            st.markdown("### Покадровое сравнение")

            col1, col2 = st.columns([1, 1])
            with col1:
                error_thresh = st.slider("Порог несоответствия, °", 0.0, 30.0, 12.0, 0.5)
            with col2:
                show_only_bad = st.checkbox("Показывать только проблемные кадры", value=False)

            frame_candidates = [
                i for i in range(aligned_len)
                if (not show_only_bad) or (per_frame_err[i] >= error_thresh)
            ]

            if not frame_candidates:
                st.success("Нет кадров, где ошибка выше порога — техника близка к эталону.")
            else:
                i = st.slider(
                    "Кадр (по траектории выравнивания)",
                    0,
                    len(frame_candidates) - 1,
                    0,
                    1,
                )
                idx = frame_candidates[i]
                fu = A["idx_user"][idx]
                fr = A["idx_ref"][idx]
                fu = max(0, min(fu, len(user_frames) - 1))
                fr = max(0, min(fr, len(ref_frames) - 1))

                st.write(
                    f"Кадр выравнивания: **{idx+1} / {aligned_len}**  "
                    f"(кадр пользователя: {fu+1}, кадр эталона: {fr+1})"
                )
                st.write(
                    f"Средняя ошибка по углам на этом кадре: **{per_frame_err[idx]:.1f}°** "
                    f"(порог {error_thresh:.1f}°)"
                )

                uf = draw_skeleton(user_frames[fu].copy(), A["user_landmarks_raw"][fu])
                rf = draw_skeleton(ref_frames[fr].copy(),  A["ref_landmarks_raw"][fr])
                combo = make_side_by_side(uf, rf, per_frame_err[idx] >= error_thresh, per_frame_err[idx])

                st.image(
                    cv2.cvtColor(combo, cv2.COLOR_BGR2RGB),
                    caption="Пользователь (слева) vs Эталон (справа)",
                )

# ======================================================================
#                        ВКЛАДКА: РЕДАКТОР ЭЛЕМЕНТОВ
# ======================================================================
with tab_editor:
    st.markdown("### Редактор элементов")

    cfg = load_config()
    ids = list(cfg.keys())
    choice = st.selectbox(
        "Выберите элемент для редактирования или создайте новый",
        ["<Новый элемент>"] + ids,
    )

    is_new = choice == "<Новый элемент>"

    if is_new:
        element_id = st.text_input("Идентификатор элемента (латиница, без пробелов)", value="")
        base = {
            "title": "",
            "reference_video": "",
            "important_joints": {},
            "tips_thresholds_deg": {"minor": 8, "major": 18},
        }
    else:
        element_id = choice
        base = cfg[element_id]

    title = st.text_input("Название элемента (будет видно в списке)", value=base.get("title", ""))

    st.write("Текущий путь к эталонному видео (относительно корня проекта):")
    st.code(base.get("reference_video", ""), language="text")

    uploaded_ref = st.file_uploader(
        "Загрузить/обновить эталонное видео (опционально)",
        type=["mp4", "avi", "mov", "mkv"],
        key="ref_upload_editor",
    )

    st.markdown("**Важность суставов (JSON, опционально)**")
    default_joints = json.dumps(base.get("important_joints", {}), ensure_ascii=False, indent=2)
    joints_text = st.text_area("Пример: {\"hip\": 2.0, \"knee\": 2.0}", value=default_joints, height=140)

    st.markdown("**Пороги для советов (JSON, опционально)**")
    default_thr = json.dumps(base.get("tips_thresholds_deg", {"minor": 8, "major": 18}),
                             ensure_ascii=False, indent=2)
    thresholds_text = st.text_area("Пример: {\"minor\": 8, \"major\": 18}", value=default_thr, height=100)

    save_btn = st.button("Сохранить элемент", type="primary", key="save_element_btn")

    if save_btn:
        cfg = load_config()  # перечитать на случай параллельных изменений

        if is_new:
            if not element_id:
                st.error("Укажите идентификатор элемента.")
            elif element_id in cfg:
                st.error("Элемент с таким идентификатором уже существует.")
            else:
                ok = True
                # обрабатываем JSON-поля
                try:
                    joints = json.loads(joints_text) if joints_text.strip() else {}
                except Exception as e:
                    st.error(f"Ошибка в JSON 'Важность суставов': {e}")
                    ok = False
                try:
                    thresholds = json.loads(thresholds_text) if thresholds_text.strip() else {"minor": 8, "major": 18}
                except Exception as e:
                    st.error(f"Ошибка в JSON 'Пороги': {e}")
                    ok = False

                ref_rel = base.get("reference_video", "")
                if uploaded_ref is not None:
                    data = uploaded_ref.read()
                    ref_name = f"{element_id}_{uploaded_ref.name}"
                    ref_path = REF_DIR / ref_name
                    with open(ref_path, "wb") as f:
                        f.write(data)
                    ref_rel = str(ref_path.as_posix())

                if ok:
                    if not ref_rel:
                        st.warning("Эталонное видео не указано. Вы сможете добавить его позже.")
                    cfg[element_id] = {
                        "title": title or element_id,
                        "reference_video": ref_rel,
                        "important_joints": joints,
                        "tips_thresholds_deg": thresholds,
                    }
                    save_config(cfg)
                    st.success(f"Элемент '{element_id}' сохранён.")
                    st.session_state.selected_element = element_id
        else:
            # редактирование существующего
            ok = True
            try:
                joints = json.loads(joints_text) if joints_text.strip() else {}
            except Exception as e:
                st.error(f"Ошибка в JSON 'Важность суставов': {e}")
                ok = False
            try:
                thresholds = json.loads(thresholds_text) if thresholds_text.strip() else {"minor": 8, "major": 18}
            except Exception as e:
                st.error(f"Ошибка в JSON 'Пороги': {e}")
                ok = False

            ref_rel = base.get("reference_video", "")
            if uploaded_ref is not None:
                data = uploaded_ref.read()
                ref_name = f"{element_id}_{uploaded_ref.name}"
                ref_path = REF_DIR / ref_name
                with open(ref_path, "wb") as f:
                    f.write(data)
                ref_rel = str(ref_path.as_posix())

            if ok:
                cfg[element_id] = {
                    "title": title or element_id,
                    "reference_video": ref_rel,
                    "important_joints": joints,
                    "tips_thresholds_deg": thresholds,
                }
                save_config(cfg)
                st.success(f"Изменения для элемента '{element_id}' сохранены.")
