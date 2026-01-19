import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.profiles import load_profile
from src.standardize import standardize_channels
from src.lmdb_reader import read_lmdb_index, read_lmdb_record, subject_to_indices

st.set_page_config(page_title="EEG Unified Pipeline", layout="wide")
st.title("EEG Unified Pipeline — Standardization + LMDB Preview")

# ---- Profile
st.sidebar.header("Profile")
profile_path = st.sidebar.selectbox("Channel profile", ["profiles/10_20_19.yaml"])
profile = load_profile(profile_path)

tab1, tab2 = st.tabs(["LMDB sample", "Fake sample"])

# ===================== TAB 1: LMDB =====================
with tab1:
    st.subheader("Load a real sample from LMDB")

    lmdb_path = st.text_input(
        "LMDB path",
        value="",
        placeholder="/projects/.../TUEG_*.lmdb",
    )

    if lmdb_path.strip():
        try:
            idx = read_lmdb_index(lmdb_path)
            st.success(f"Loaded LMDB index: total_len={idx.total_len}, n_subjects={len(idx.subject_ids)}")
        except Exception as e:
            st.error(f"Failed to read LMDB index: {e}")
            st.stop()

        # subject selection (if ranges exist)
        subject_mode = st.radio("Pick by", ["global index", "subject"], horizontal=True)

        if subject_mode == "global index" or not idx.subject_ids:
            global_i = st.number_input("Global index", min_value=0, max_value=max(0, idx.total_len - 1), value=0, step=1)
            rec = read_lmdb_record(lmdb_path, int(global_i))

        else:
            sid = st.selectbox("Subject ID", idx.subject_ids)
            indices = subject_to_indices(idx, sid)
            if not indices:
                st.warning("No ranges for this subject in LMDB metadata.")
                st.stop()

            which = st.slider("Pick window within subject", 0, len(indices) - 1, 0)
            global_i = indices[which]
            rec = read_lmdb_record(lmdb_path, int(global_i))

        # extract fields (based on Yanic record)
        x_raw = rec["x"]                 # numpy (C,T)
        ch_names = rec.get("ch_names", [])
        y = int(rec.get("label", -1))
        sfreq = float(rec.get("sfreq", float("nan")))
        subject_id = rec.get("subject_id", "")
        path = rec.get("path", "")

        # standardize
        x_ref, mask, chan_ids = standardize_channels(x_raw, ch_names, profile)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Metadata")
            st.write("global_idx:", int(global_i))
            st.write("dataset path:", path)
            st.write("subject_id:", subject_id)
            st.write("label:", y)
            st.write("sfreq:", sfreq)

            st.subheader("Shapes")
            st.write("raw x:", np.asarray(x_raw).shape)
            st.write("ref x:", x_ref.shape)

            matched = [profile.channels[i] for i, v in enumerate(mask) if v]
            missing = [profile.channels[i] for i, v in enumerate(mask) if not v]
            st.subheader("Matched / Missing")
            st.write("matched:", matched if matched else "None")
            st.write("missing:", missing if missing else "None")
            st.write("mask sum:", int(mask.sum()))

        with col2:
            st.subheader("Plot")
            ch_to_plot = st.selectbox("Channel (reference)", profile.channels, index=0, key="lmdb_plot_ch")
            ref_i = profile.index[ch_to_plot]

            fig = plt.figure()
            plt.plot(x_ref[ref_i])
            plt.title(f"{ch_to_plot} (present={bool(mask[ref_i])})")
            plt.xlabel("t")
            plt.ylabel("amplitude")
            st.pyplot(fig)

    else:
        st.info("Paste an LMDB path to start.")

# ===================== TAB 2: Fake sample =====================
with tab2:
    st.subheader("Fake sample (for sanity check)")
    fake_chs = st.multiselect(
        "Input channel names (raw)",
        ["Fp1", "Fp2", "F3", "T3", "T4", "Cz", "O2", "Pz", "X1"],
        default=["Fp1", "T3", "Cz", "O2"],
        key="fake_chs"
    )
    T = st.slider("Timepoints (T)", min_value=200, max_value=5000, value=1000, step=100, key="fake_T")
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1, key="fake_seed")

    rng = np.random.default_rng(int(seed))
    x_raw = rng.standard_normal((len(fake_chs), T)).astype(np.float32)

    x_ref, mask, chan_ids = standardize_channels(x_raw, fake_chs, profile)

    st.write("Raw x:", x_raw.shape, "→ Ref x:", x_ref.shape)
    st.write("mask sum:", int(mask.sum()))

    matched = [profile.channels[i] for i, v in enumerate(mask) if v]
    missing = [profile.channels[i] for i, v in enumerate(mask) if not v]
    coverage = float(mask.sum()) / float(len(profile.channels)) if len(profile.channels) else 0.0

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Matched channels (present)**")
        st.write(matched if matched else "None")

    with colB:
        st.markdown("**Missing channels (padded)**")
        st.write(missing if missing else "None")

    st.markdown(f"**Coverage:** {mask.sum()}/{len(profile.channels)} = {coverage*100:.1f}%")
