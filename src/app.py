import streamlit as st
import os
from model import load_clip_model
from image_processing import compute_similarity
from utils import is_valid_folder, save_matching_images

def main():
    model, processor = load_clip_model()

    st.title("Image Finder with Description Matching")
    st.write("This tool scans a folder and finds images matching the provided description.")
    
    #input description/folder
    description = st.text_input("Enter a description for the image search:")
    image_folder = st.text_input("Enter the folder path containing images:")
    
    #search button
    if st.button("Find Images"):
        if not description or not image_folder:
            st.warning("Please provide both a description and a valid folder path.")
        elif not is_valid_folder(image_folder):
            st.error("The specified folder does not exist.")
        else:
            with st.spinner("Processing images..."):
                results = compute_similarity(model, processor, description, image_folder)
            st.success("Search completed!")
            
            #display results
            if results:
                st.write(f"Found {len(results)} images. Displaying matches in descending order of similarity:")
                for filename, similarity in results[:10]:
                    st.write(f"Image: {filename}, Similarity: {similarity:.2f}")
                    image_path = os.path.join(image_folder, filename)
                    st.image(image_path, width=300)
            else:
                st.write("No matching images found.")

if __name__ == "__main__":
    main()
