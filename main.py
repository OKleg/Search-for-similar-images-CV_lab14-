try:
    import streamlit as st
    import cv2
    import numpy as np
    import os
    import sys
    import pandas as pd
    import pickle
    from sklearn.neighbors import NearestNeighbors
    import ast
    import torch
    import clip
    from PIL import Image
    from io import StringIO, BytesIO
    print("All Modules Loaded")
except Exception as e:
    print("Some Modules are Missing  : {}", format(e))
st.title("Hi")

STYLE = """
<style>
img {
    max-with:100%;
}
<style>
"""

class FileUpload(object):

    def __init__(self, fileTypes = ['png', 'jpg']):
        self.fileTypes = fileTypes

    def run(self):
        """
            Upload File on Streamlit code
            :return:
        """
        st.info(__doc__)
        st.markdown(STYLE, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload file", type=self.fileTypes)
        show_file = st.empty()
        if not uploaded_file:
            show_file.info("Please, Upload a image: {}".format(' '.join(self.fileTypes)))
            return
        content = uploaded_file.getvalue()
        if isinstance(uploaded_file,BytesIO):
            show_file.image(uploaded_file)
if __name__ == "__main__":
    helper = FileUpload()
    helper.run()
    print(torch.cuda.is_available()) # Должно быть True
    