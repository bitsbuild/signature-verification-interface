import streamlit as st
st.title("Result")
if "match" in st.session_state:
    if st.session_state["match"] == True:
        st.write("Genuine")
    else:
        st.write("Forged")