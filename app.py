import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
    layout="wide"
)

# =============================
# THEME TOGGLE (TOP RIGHT)
# =============================
col1, col2 = st.columns([8, 1])
with col2:
    dark_mode = st.toggle("🌙 Dark Mode")

if dark_mode:
    st.markdown("""
        <style>
        .stApp {background-color: #0e1117; color: white;}
        section[data-testid="stSidebar"] {background-color: #161b22 !important;}
        h1, h2, h3, h4 {color: white !important;}
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp {background-color: #f7f9fc;}
        section[data-testid="stSidebar"] {background-color: #ffffff;}
        </style>
    """, unsafe_allow_html=True)

# =============================
# HEADER
# =============================
st.title("🛣️ Road Damage Detection System")
st.markdown("### YOLOv8 Detection – Multi-Class Road Damage")

# =============================
# SIDEBAR
# =============================
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3)
st.sidebar.markdown("---")
st.sidebar.info(
    "Model: YOLOv8 Detect\n\n"
    "Classes:\n"
    "- Longitudinal Crack\n"
    "- Transverse Crack\n"
    "- Alligator Crack\n"
    "- Other Corruption\n"
    "- Pothole"
)

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =============================
# FILE UPLOADER
# =============================
uploaded_files = st.file_uploader(
    "Upload Image(s) or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
    accept_multiple_files=True
)

# =============================
# PROCESS FILES
# =============================
if uploaded_files:

    for file in uploaded_files:

        file_type = file.name.split(".")[-1].lower()
        st.markdown("---")
        st.subheader(f"📁 {file.name}")

        # =====================================================
        # IMAGE PROCESSING
        # =====================================================
        if file_type in ["jpg", "jpeg", "png"]:

            image = Image.open(file).convert("RGB")

            with st.spinner("Processing image..."):
                results = model(image, conf=confidence)
                annotated_image = results[0].plot()

            st.image(annotated_image, use_column_width=True)

            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                class_ids = boxes.cls.cpu().numpy().astype(int)

                # Initialize all classes with 0
                class_counts = {name: 0 for name in model.names.values()}

                for cid in class_ids:
                    class_name = model.names[cid]
                    class_counts[class_name] += 1

                st.markdown("### 📊 Damage Count (Image)")

                cols = st.columns(len(class_counts))
                for col, (name, count) in zip(cols, class_counts.items()):
                    col.metric(name, count)
            else:
                st.warning("No road damage detected.")

            # Download image
            _, buffer = cv2.imencode(".png", annotated_image)
            st.download_button(
                label="⬇ Download Processed Image",
                data=buffer.tobytes(),
                file_name=f"processed_{file.name}",
                mime="image/png"
            )

        # =====================================================
        # VIDEO PROCESSING WITH TRACKING
        # =====================================================
        elif file_type in ["mp4", "avi", "mov"]:

            st.info("Processing video with tracking... ⏳")

            # Save uploaded file temporarily
            suffix = "." + file_type
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tfile.write(file.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            output_path = os.path.join(
                tempfile.gettempdir(),
                f"processed_{file.name}"
            )

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Unique tracking storage
            unique_ids = set()
            class_wise_ids = {name: set() for name in model.names.values()}

            frame_placeholder = st.empty()
            progress_bar = st.progress(0)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=confidence
                )

                annotated_frame = results[0].plot()
                boxes = results[0].boxes

                if boxes.id is not None:
                    ids = boxes.id.cpu().numpy().astype(int)
                    class_ids = boxes.cls.cpu().numpy().astype(int)

                    for obj_id, cid in zip(ids, class_ids):
                        unique_ids.add(obj_id)

                        class_name = model.names[cid]
                        class_wise_ids[class_name].add(obj_id)

                out.write(annotated_frame)

                # Continuous UI update
                frame_placeholder.image(
                    annotated_frame,
                    channels="BGR",
                    use_column_width=True
                )

                frame_count += 1
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            out.release()
            progress_bar.empty()

            st.success("Video processing completed ✅")
            st.success(f"Total Frames Processed: {frame_count}")
            st.success(f"Total Unique Damages Detected: {len(unique_ids)}")

            st.markdown("### 📊 Class-wise Unique Counts (Video)")

            cols = st.columns(len(class_wise_ids))
            for col, (class_name, ids) in zip(cols, class_wise_ids.items()):
                col.metric(class_name, len(ids))

            # Display processed video
            with open(output_path, "rb") as f:
                video_bytes = f.read()

            st.video(video_bytes)

            st.download_button(
                label="⬇ Download Processed Video",
                data=video_bytes,
                file_name=f"processed_{file.name}",
                mime="video/mp4"
            )