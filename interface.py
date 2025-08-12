import streamlit as st
st.title("Signature Verification")
upload_file_0 = st.file_uploader("Upload Original Signature",type=["jpg","jpeg","png"])
upload_file_1 = st.file_uploader("Upload Signature To Be Verified",type=["jpg","jpeg","png"])