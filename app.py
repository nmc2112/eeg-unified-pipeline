import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.profiles import load_profile
from src.standardize import standardize_channels

st.set_page_config(page_title="EEG Unified Pipeline", layout="wide")
st.title("EEG Unified Pipeline — Channel Standardization Preview")

# ---- Sidebar: select profile
st.sidebar.header("Profile")
profile_path = st.sidebar.selectbox(
    "Channel profile",
    ["profiles/10_20_19.yaml"],
)
profile = load_profile(profile_path)

st.sidebar.header("Fake sample generator")
fake_chs = st.sidebar.multiselect(
    "Input channel names (raw)",
    ["Fp1", "Fp2", "F3", "T3", "T4", "Cz", "O2", "Pz", "X1"],
    default=["Fp1", "T3", "Cz", "O2"],
)
T = st.sidebar.slider("Timepoints (T)", min_value=200, max_value=5000, value=1000, step=100)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)

# ---- Create fake sample
rng = np.random.default_rng(int(seed))
x_raw = rng.standard_normal((len(fake_chs), T)).astype(np.float32)

# ---- Standardize
x_ref, mask, chan_ids = standardize_channels(x_raw, fake_chs, profile)

# ---- Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Shapes")
    st.write("Raw x:", x_raw.shape)
    st.write("Ref x:", x_ref.shape)
    st.write("Profile:", f"{profile.name} ({len(profile.channels)} channels)")

    matched = [profile.channels[i] for i, v in enumerate(mask) if v]
    missing = [profile.channels[i] for i, v in enumerate(mask) if not v]

    st.subheader("Matched / Missing")
    st.write("Matched:", matched if matched else "None")
    st.write("Missing:", missing if missing else "None")

    st.subheader("Mask / chan_ids")
    st.write("mask sum:", int(mask.sum()))
    st.write("chan_ids (first 30):", chan_ids[:30].tolist())

with col2:
    st.subheader("Plot one reference channel")
    ch_to_plot = st.selectbox("Channel", profile.channels, index=0)
    idx = profile.index[ch_to_plot]

    fig = plt.figure()
    plt.plot(x_ref[idx])
    plt.title(f"{ch_to_plot} (present={bool(mask[idx])})")
    plt.xlabel("t")
    plt.ylabel("amplitude (a.u.)")
    st.pyplot(fig)

st.info("Next step: replace fake sample with real LMDB sample loader.")
