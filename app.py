import streamlit as st

st.set_page_config(page_title="EEG Unified Pipeline", layout="wide")
st.title("EEG Unified Pipeline - Demo UI")

st.sidebar.header("Controls")
dataset = st.sidebar.selectbox("Dataset", ["TUEG", "TUAB", "SLEEPEDF", "SEEDV", "PHYSIONETMI", "FACED", "BCI2A"])
model = st.sidebar.selectbox("Model", ["EEGPT", "MyModel (placeholder)"])

st.write("Selected dataset:", dataset)
st.write("Selected model:", model)
st.info("Next: connect this UI to LMDB loader + plotting + inference.")
