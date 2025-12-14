# -*- coding: utf-8 -*-
import os
import json
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from app.pose_extractor import extract_pose_from_video, get_video_frames
from app.preprocessing import normalize_landmarks, compute_angles_sequence, smooth_series
from app.dtw_utils import stack_features, align_by_dtw
from app.visualization import draw_skeleton_pil, make_side_by_side_pil

# ---------------------------- CONFIG ----------------------------

CONFIG_PATH = Path("app/elements_config.json")
REF_DIR = Path("references")
REF_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="AI-коуч", layout="wide")
st.title("🤸 AI-коуч: покадровое сравнение техники")

def load_config():
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

# ---------------------------- SESSION ----------------------------

if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "user_video_path" not in st.session_state:
    st.session_state.user_video_path = None

# ---------------------------- UTILS ----------------------------

def save_temp_video(uploaded_file):
    data = uploaded_file.read()
    tmp_dir = Path(tempfile.gettempdir()) / "ai_coach"
    tmp_dir.mkdir(exist_ok=True)
    path = tmp_dir / uploaded_file.name
    with open(path, "wb") as f:
        f.write(data)
    return str(path)

@st.cache_data(show_spinner=False)
def analyze_video(user_path, ref_path):
    user_seq = extract_pose_from_video(user_path)
    ref_seq = extract_pose_from_video(ref_path)

    user_norm = normalize_landmarks(user_seq.landmarks)
    ref_norm = normalize_landmarks(ref_seq.landmarks)

    user_angles = smooth_series(compute_angles_sequence(user_norm), 11, 3)
    ref_angles = smooth_series(compute_angles_sequence(ref_norm), 11, 3)

    keys = list(user_angles.keys())[:6]
    user_feat = stack_features(user_angles, keys)
    ref_feat = stack_features(ref_angles, keys)

    _, _, path = align_by_dtw(user_feat, ref_feat)
    idx_user = [p[0] for p in path]
    idx_ref = [p[1] for p in path]

    return {
        "user_seq": user_seq,
        "ref_seq": ref_seq,
        "idx_user": idx_user,
        "idx_ref": idx_ref,
        "user_angles": user_angles,
        "ref_angles": ref_angles,
    }

# ---------------------------- UI ----------------------------

cfg = load_config()
if not cfg:
    st.warning("Добавьте элементы в elements_config.json")
    st.stop()

el = st.selectbox(
    "Выберите элемент",
    list(cfg.keys()),
    format_func=lambda k: cfg[k]["title"]
)

user_file = st.file_uploader("Загрузите видео пользователя", type=["mp4", "mov", "avi"])
if st.button("Анализировать"):
    if not user_file:
        st.error("Загрузите видео")
        st.stop()

    user_path = save_temp_video(user_file)
    ref_path = cfg[el]["reference_video"]

    with st.spinner("Анализ видео..."):
        st.session_state.analysis = analyze_video(user_path, ref_path)
        st.session_state.user_video_path = user_path

if st.session_state.analysis is None:
    st.info("Загрузите видео и нажмите «Анализировать»")
    st.stop()

A = st.session_state.analysis

user_frames = get_video_frames(st.session_state.user_video_path)[1]
ref_frames = get_video_frames(cfg[el]["reference_video"])[1]

aligned_len = len(A["idx_user"])

st.subheader("Покадровое сравнение")
frame_id = st.slider("Кадр выравнивания", 0, aligned_len - 1, 0)

fu = A["idx_user"][frame_id]
fr = A["idx_ref"][frame_id]

img_user = draw_skeleton_pil(
    user_frames[fu],
    A["user_seq"].landmarks[fu]
)
img_ref = draw_skeleton_pil(
    ref_frames[fr],
    A["ref_seq"].landmarks[fr]
)

combo = make_side_by_side_pil(img_user, img_ref)
st.image(combo, caption="Пользователь (слева) / Эталон (справа)")
