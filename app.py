import streamlit as st
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import torch
import pandas as pd
from nets import CLIP, tokenize
import os

def label2text2token(label, labmap=None):
    # 转换标签为整数
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy().tolist()
    elif isinstance(label, list):
        label = label
    # 定义医学标签映射
    label_map = {
        0: "invasive adenocarcinoma",
        1: "minimally invasive adenocarcinoma",
        2: "adenocarcinoma in situ",
        3: "adenocarcinoma in situ"
    }
    if labmap == 'LPCD':
        label_map = {
            0: "adenocarcinoma", # A
            1: "squamous cell carcinoma", # G
            2: "small cell carcinoma", # B
            # 3: "adenocarcinoma in situ"
        }
    # 生成医学描述文本
    texts = [
        f"A pulmonary nodule showing histologic features of {label_map[l]}."
        for l in label
    ]
    # 分词处理
    encoding = tokenize(texts)
    return encoding

st.set_page_config(page_title="NIfTI 裁剪浏览器", layout="wide")
st.title("🧠 肺腺癌预测模型")

uploaded_file = st.file_uploader("上传 .nii.gz 文件", type=["nii.gz"])
st.markdown("### 📄 上传临床信息 CSV 文件")
clinical_csv = st.file_uploader("上传包含 'bid' 和 27个f列 的 CSV 文件", type="csv")
clinical_tensor = None
pid = os.path.basename(uploaded_file.name).replace(".nii.gz", "") if uploaded_file else None

if clinical_csv is not None and pid:
    df = pd.read_csv(clinical_csv)
    if 'pid' not in df.columns:
        st.error("CSV中缺少 'pid' 列")
    else:
        matched_row = df[df['pid'] == pid]
        if not matched_row.empty:
            # 提取以f开头的列
            # clinical_values = matched_row.filter(regex='^f').values.flatten().tolist()
            st.success(f"✅ 成功匹配临床信息：共 {len(matched_row)} 项")
            st.write(matched_row.T)
            try:
                bid = st.selectbox("📦 选择匹配的临床信息", matched_row['bid'])
                clinical_values = matched_row[matched_row['bid'] == bid].filter(regex='^f').values.flatten().tolist()
                clinical_tensor = torch.tensor(clinical_values, dtype=torch.float32).unsqueeze(0)
            except:
                st.warning("⚠️ 无法转换为Tensor，请检查数值")
        else:
            st.warning(f"⚠️ 未找到与 bid '{pid}' 匹配的临床信息")

model_choices = [f for f in os.listdir("./models") if f.endswith(".pt")]
model_name = st.selectbox("📦 选择模型文件 (.pt)", model_choices) if model_choices else None

# 初始化状态
if "use_crop" not in st.session_state:
    st.session_state.use_crop = False
if "model" not in st.session_state:
    st.session_state.model = None

if model_name and st.session_state.model is None:
    model_path = os.path.join("./models", model_name)
    weights = torch.load(model_path, map_location='cpu')
    model = CLIP(embed_dim=256, model_depth=18, clinical_dim=27,
                 context_length=77, vocab_size=49408,
                 transformer_width=128, transformer_heads=4, transformer_layers=2)
    model.load_state_dict(weights['cls_model'])
    model.eval()
    st.session_state.model = model
    st.success(f"✅ 模型 {model_name} 加载成功！")

if uploaded_file and clinical_csv:
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    image = sitk.ReadImage(tmp_path)
    full_volume = sitk.GetArrayFromImage(image).transpose(1, 2, 0)  # shape: (z, y, x)
    # print(full_volume.shape)
    shape_x, shape_y, shape_z = full_volume.shape
    center_values = [shape_x//2, shape_y//2, shape_z//2]
    if bid:
        # print(matched_row[matched_row['bid'] == bid]['bbox'].values[0])
        center_values = eval(matched_row[matched_row['bid'] == bid]['bbox'].values[0])
        center_values = center_values[:3]
        print(center_values)

    st.markdown("### 🧭 手动设置裁剪中心点坐标（z, y, x）")
    cz, cy, cx = st.columns(3)
    with cz:
        z = st.number_input("Z（轴向）", min_value=0, max_value=shape_z - 1, value=center_values[2], step=1)
    with cy:
        y = st.number_input("Y（冠状）", min_value=0, max_value=shape_y - 1, value=center_values[1], step=1)
    with cx:
        x = st.number_input("X（矢状）", min_value=0, max_value=shape_x - 1, value=center_values[0], step=1)

    center = (int(z), int(y), int(x))
    size = 32
    half = size // 2
    z1, z2 = max(z - half, 0), min(z + half, shape_z)
    y1, y2 = max(y - half, 0), min(y + half, shape_y)
    x1, x2 = max(x - half, 0), min(x + half, shape_x)
    cropped_volume = full_volume[x1:x2, y1:y2, z1:z2]

    col1, col2 = st.columns(2)
    with col1:
        if st.button("显示裁剪图像 🪚"):
            st.session_state.use_crop = True
    with col2:
        if st.button("显示原图 🧠"):
            st.session_state.use_crop = False

    volume = cropped_volume if st.session_state.use_crop else full_volume
    index = st.slider("切片索引", 0, min(volume.shape) - 1, min(volume.shape) // 2)
    axial = volume[index, :, :] if volume.shape[0] > index else volume[-1, :, :]
    coronal = volume[:, index, :] if volume.shape[1] > index else volume[:, -1, :]
    sagittal = volume[:, :, index] if volume.shape[2] > index else volume[:, :, -1]

    cols = st.columns(3)
    for col, (title, img) in zip(cols, [("轴向", sagittal), ("冠状", coronal), ("矢状", axial)]):
        with col:
            st.markdown(f"### {title}")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            st.pyplot(fig)

    st.markdown("### 🏷️ 输入类别文本（用于对比）")
    label_input = st.text_input("类别文本（用英文逗号分隔）", value="invasive adenocarcinoma,minimally invasive adenocarcinoma,adenocarcinoma in situ")
    class_labels = [s.strip() for s in label_input.split(",")]
    text_tokens = label2text2token(label=[0, 1, 2])
    # try:
    #     # text_tokens = tokenize(class_labels)
    #     text_tokens = label2text2token()
    # except Exception as e:
    #     st.error(f"❌ 文本编码失败: {e}")
    #     text_tokens = None

    if st.button("🔍 执行预测"):
        if st.session_state.model is None:
            st.error("请先上传模型")
        elif clinical_tensor is None:
            st.error("临床特征输入有误")
        elif text_tokens is None:
            st.error("类别文本输入有误")
        else:
            with torch.no_grad():
                ct_norm = np.clip(cropped_volume, -600, 900)
                ct_norm = (ct_norm + 600) / 1500
                ct_tensor = torch.tensor(ct_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                logits_per_image, _ = st.session_state.model(ct_tensor, clinical_tensor, text_tokens)
                probs = torch.softmax(logits_per_image, dim=1).squeeze().numpy()
                pred = int(np.argmax(probs))
                st.success(f"🎯 预测结果：类别 {pred} → {class_labels[pred]}")
                st.write(f"📊 概率分布：{dict(zip(class_labels, probs.round(4).tolist()))}")
