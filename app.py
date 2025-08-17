# app.py
"""
GlamCam — Minimal working Streamlit live virtual makeup prototype.

Run:
    pip install streamlit opencv-python mediapipe numpy pillow streamlit-webrtc av
    streamlit run app.py

Notes:
- If you see slowness, close other heavy apps or lower webcam resolution.
- If the camera doesn't start, refresh the page and allow camera permission.
"""

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase, RTCConfiguration

st.set_page_config(page_title="GlamCam — Live Virtual Try-On", layout="wide")

# -------------------------
# Utility: convert hex -> RGB tuple
# -------------------------
def hex_to_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # (R, G, B)

# -------------------------
# Utility: smooth polygon mask
# -------------------------
def polygon_mask(shape, polygon_pts, blur_ksize=41):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if polygon_pts is None or len(polygon_pts) < 3:
        return mask.astype(float)
    cv2.fillPoly(mask, [np.array(polygon_pts, dtype=np.int32)], 255)
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    # keep k reasonable relative to image
    k = min(k, max(1, (min(shape[:2]) // 2) | 1))
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return (mask.astype(np.float32) / 255.0)

def blend_color(image_rgb, color_rgb, mask, opacity=0.6):
    """Blend color_rgb (tuple R,G,B) onto image_rgb using mask (HxW floats 0..1)."""
    overlay = np.full_like(image_rgb, color_rgb, dtype=np.uint8)
    mask_3 = np.stack([mask, mask, mask], axis=2)
    blended = (image_rgb.astype(np.float32) * (1 - mask_3 * opacity) + overlay.astype(np.float32) * (mask_3 * opacity))
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended

# -------------------------
# Face landmark index sets (approximate)
# -------------------------
mp_face_mesh = mp.solutions.face_mesh

# helper to fetch unique indices from connections
def indices_from_connections(connections):
    s = set()
    for a, b in connections:
        s.add(a); s.add(b)
    return sorted(s)

LIPS_IDX = indices_from_connections(mp_face_mesh.FACEMESH_LIPS)
LEFT_EYE_IDX = indices_from_connections(mp_face_mesh.FACEMESH_LEFT_EYE)
RIGHT_EYE_IDX = indices_from_connections(mp_face_mesh.FACEMESH_RIGHT_EYE)
FACE_OVAL_IDX = indices_from_connections(mp_face_mesh.FACEMESH_FACE_OVAL)
LEFT_CHEEK_APPROX = [234, 93, 132]   # heuristic points (may be adjusted)
RIGHT_CHEEK_APPROX = [454, 323, 361] # heuristic points

# -------------------------
# Video transformer
# -------------------------
class MakeupTransformer(VideoTransformerBase):
    def __init__(self, config):
        """
        config: dict with keys:
          - foundation_color (hex str)
          - foundation_opacity (float 0..1)
          - lipstick_color (hex str)
          - lipstick_opacity (float 0..1)
          - eyeshadow_color (hex str)
          - eyeshadow_opacity (float 0..1)
          - blush_color (hex str)
          - blush_opacity (float 0..1)
        """
        self.config = config
        # create face mesh detector once
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                              max_num_faces=1,
                                              refine_landmarks=True,
                                              min_detection_confidence=0.5,
                                              min_tracking_confidence=0.5)

    def transform(self, frame):
        # frame is an av.VideoFrame-like object
        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return img_bgr  # return original BGR frame if no face detected

        # Work on RGB image for nicer blending calculations
        out_rgb = img_rgb.copy()

        # Use first face only
        face_landmarks = results.multi_face_landmarks[0].landmark

        # FOUNDATION (face oval)
        face_pts = [(int(lm.x * w), int(lm.y * h)) for lm in (face_landmarks[i] for i in FACE_OVAL_IDX if i < len(face_landmarks))]
        foundation_mask = polygon_mask(out_rgb.shape, face_pts, blur_ksize=101)
        foundation_rgb = hex_to_rgb(self.config.get("foundation_color", "#d0b08a"))
        out_rgb = blend_color(out_rgb, foundation_rgb, foundation_mask, opacity=self.config.get("foundation_opacity", 0.6))

        # LIPSTICK
        lip_pts = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in LIPS_IDX if i < len(face_landmarks)]
        if len(lip_pts) >= 3 and self.config.get("lipstick_opacity", 0.8) > 0:
            lip_mask = polygon_mask(out_rgb.shape, lip_pts, blur_ksize=21)
            lipstick_rgb = hex_to_rgb(self.config.get("lipstick_color", "#b1223b"))
            out_rgb = blend_color(out_rgb, lipstick_rgb, lip_mask, opacity=self.config.get("lipstick_opacity", 0.8))

        # EYESHADOW (left & right)
        left_eye_pts = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in LEFT_EYE_IDX if i < len(face_landmarks)]
        right_eye_pts = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in RIGHT_EYE_IDX if i < len(face_landmarks)]
        eyeshadow_rgb = hex_to_rgb(self.config.get("eyeshadow_color", "#6e4b9b"))
        if self.config.get("eyeshadow_opacity", 0.5) > 0:
            if len(left_eye_pts) >= 3:
                left_mask = polygon_mask(out_rgb.shape, left_eye_pts, blur_ksize=21)
                out_rgb = blend_color(out_rgb, eyeshadow_rgb, left_mask, opacity=self.config.get("eyeshadow_opacity", 0.5))
            if len(right_eye_pts) >= 3:
                right_mask = polygon_mask(out_rgb.shape, right_eye_pts, blur_ksize=21)
                out_rgb = blend_color(out_rgb, eyeshadow_rgb, right_mask, opacity=self.config.get("eyeshadow_opacity", 0.5))

        # BLUSH (approximate using cheek anchor points)
        blush_rgb = hex_to_rgb(self.config.get("blush_color", "#ff9fb3"))
        if self.config.get("blush_opacity", 0.25) > 0:
            # build small circular masks around chosen landmark indices
            cheek_mask = np.zeros((h, w), dtype=np.uint8)
            for idx in LEFT_CHEEK_APPROX + RIGHT_CHEEK_APPROX:
                if idx < len(face_landmarks):
                    cx = int(face_landmarks[idx].x * w)
                    cy = int(face_landmarks[idx].y * h)
                    r = max(12, min(h, w) // 12)
                    cv2.circle(cheek_mask, (cx, cy), r, 255, -1)
            # blur and normalize
            k = 51 if (51 % 2 == 1) else 51 + 1
            cheek_mask = cv2.GaussianBlur(cheek_mask, (k, k), 0)
            cheek_mask = (cheek_mask.astype(np.float32) / 255.0)
            out_rgb = blend_color(out_rgb, blush_rgb, cheek_mask, opacity=self.config.get("blush_opacity", 0.25))

        # EYELINER: draw thin lines along eyelid landmarks (dark color)
        try:
            eyeliner_img = out_rgb.copy()
            if len(left_eye_pts) >= 3:
                pts = np.array(left_eye_pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(eyeliner_img, [pts], isClosed=False, color=(10, 10, 10), thickness=1, lineType=cv2.LINE_AA)
            if len(right_eye_pts) >= 3:
                pts = np.array(right_eye_pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(eyeliner_img, [pts], isClosed=False, color=(10, 10, 10), thickness=1, lineType=cv2.LINE_AA)
            # merge the eyeliner subtly
            out_rgb = cv2.addWeighted(out_rgb, 0.85, eyeliner_img, 0.15, 0)
        except Exception:
            pass

        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        return out_bgr

# -------------------------
# Streamlit UI
# -------------------------
st.title("✨ GlamCam — Live Virtual Makeup Try-On")

tab1, tab2 = st.tabs(["Home", "Live"])

with tab1:
    st.header("Search & Product Gallery")
    q = st.text_input("Search for a makeup item (example: matte lipstick)")
    st.write("Sample product images (images only — no prices):")
    cols = st.columns(3)
    sample_urls = [
        "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=600&q=60",
        "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=600&q=60",
        "https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9?w=600&q=60",
    ]
    for c, url in zip(cols, sample_urls):
        with c:
            # use width param (works across Streamlit versions)
            st.image(url, caption="Sample product", width=220)

with tab2:
    st.header("Live Try-On (allow camera access)")
    # Presets
    preset = st.selectbox("Preset", ["Natural", "Matte", "Dewy", "Custom"])

    if preset == "Natural":
        foundation_hex = "#d0b08a"
        lipstick_hex = "#8a1a36"
        eyeshadow_hex = "#6e4b9b"
        blush_hex = "#e88ca0"
        foundation_op = 0.55
        lipstick_op = 0.85
        eyeshadow_op = 0.5
        blush_op = 0.25
    elif preset == "Matte":
        foundation_hex = "#c9a682"
        lipstick_hex = "#7b1b2f"
        eyeshadow_hex = "#5a2f66"
        blush_hex = "#d87a90"
        foundation_op = 0.6
        lipstick_op = 0.95
        eyeshadow_op = 0.6
        blush_op = 0.3
    elif preset == "Dewy":
        foundation_hex = "#e6c9a1"
        lipstick_hex = "#a83a4a"
        eyeshadow_hex = "#8d5db6"
        blush_hex = "#f0a3b6"
        foundation_op = 0.35
        lipstick_op = 0.65
        eyeshadow_op = 0.45
        blush_op = 0.22
    else:
        # Custom defaults
        foundation_hex = st.color_picker("Foundation shade", "#d0b08a")
        lipstick_hex = st.color_picker("Lipstick", "#b1223b")
        eyeshadow_hex = st.color_picker("Eyeshadow", "#6e4b9b")
        blush_hex = st.color_picker("Blush", "#e88ca0")
        foundation_op = st.slider("Foundation intensity", 0.0, 1.0, 0.55)
        lipstick_op = st.slider("Lipstick intensity", 0.0, 1.0, 0.85)
        eyeshadow_op = st.slider("Eyeshadow intensity", 0.0, 1.0, 0.5)
        blush_op = st.slider("Blush intensity", 0.0, 1.0, 0.25)

    # If a preset (not custom) — still show the color pickers so user can tweak
    if preset != "Custom":
        foundation_hex = st.color_picker("Foundation shade", foundation_hex)
        lipstick_hex = st.color_picker("Lipstick", lipstick_hex)
        eyeshadow_hex = st.color_picker("Eyeshadow", eyeshadow_hex)
        blush_hex = st.color_picker("Blush", blush_hex)
        foundation_op = st.slider("Foundation intensity", 0.0, 1.0, float(foundation_op))
        lipstick_op = st.slider("Lipstick intensity", 0.0, 1.0, float(lipstick_op))
        eyeshadow_op = st.slider("Eyeshadow intensity", 0.0, 1.0, float(eyeshadow_op))
        blush_op = st.slider("Blush intensity", 0.0, 1.0, float(blush_op))

    # Build config dict
    config = {
        "foundation_color": foundation_hex,
        "foundation_opacity": float(foundation_op),
        "lipstick_color": lipstick_hex,
        "lipstick_opacity": float(lipstick_op),
        "eyeshadow_color": eyeshadow_hex,
        "eyeshadow_opacity": float(eyeshadow_op),
        "blush_color": blush_hex,
        "blush_opacity": float(blush_op),
    }

    # Create transformer factory (constructed with current config)
    def transformer_factory():
        return MakeupTransformer(config=config)

    # Optional RTC config (helps in some browsers / deployments)
    rtc_conf = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    webrtc_streamer(
        key="glamcam",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=transformer_factory,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=rtc_conf,
    )

st.caption("Prototype — for production-friendly client-side performance consider a React + MediaPipe JS implementation.")
