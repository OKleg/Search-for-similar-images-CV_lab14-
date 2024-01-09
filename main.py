import streamlit as st
import numpy as np
import PIL
from similar import Similar

print("All Modules Loaded")

st.title("Search for similar images")

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
        # st.info(__doc__)
        #st.markdown(STYLE, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload image", type=self.fileTypes)
        st.title("Input Image")
        show_file = st.empty()

        if not uploaded_file:
            show_file.info("Please, Upload a image: {}".format(' '.join(self.fileTypes)))
            return

        content = uploaded_file.getvalue()
        
        if uploaded_file:

            show_file.image(uploaded_file, caption='Uploaded Image.')
        cur_image = PIL.Image.open(uploaded_file)
        cur_image = np.array(cur_image)
        
        return cur_image
def change():
    print("change")
if __name__ == "__main__":
    helper = FileUpload()
    input_image = helper.run()
    k = st.number_input("Select number similar images", 1, 10, 6, 2)
    if input_image is not None:

        st.title("Similar Images")
        sim = Similar(input_image,k)
        similar_images = sim.run()
        pathes = []
        indices_on_page = []
        cols = st.columns(2)
        
        groups = []
        for i in range(0,len(similar_images),2):
            groups.append(similar_images[i:i+2])

        for group in groups:
            for i, path in enumerate(group):  
                # Remove the unwanted directory from the path

                path = path.replace('/content/drive/MyDrive', './webapp')
                cols[i].image(path,width=200,use_column_width = "auto")


        # st.image(pathes,width=200,use_column_width = "auto", caption=indices_on_page) 
    