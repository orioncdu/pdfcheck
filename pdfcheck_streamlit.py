import streamlit as st
import fitz  # PyMuPDF
from PIL import Image, ImageChops
import numpy as np
from scipy.ndimage import binary_dilation

st.title("Interactive PDF Pixel-by-Pixel Comparison")

# 1. Upload multiple files at once
uploaded_files = st.file_uploader(
    "Upload your PDF files (you can upload as many as you like):", 
    type="pdf", 
    accept_multiple_files=True
)

def compare_pdfs_pixel_by_pixel(pdf_bytes_1, pdf_bytes_2, dpi=150, dilation_iterations=2):
    doc1 = fitz.open(stream=pdf_bytes_1, filetype="pdf")
    doc2 = fitz.open(stream=pdf_bytes_2, filetype="pdf")

    if len(doc1) != len(doc2):
        st.error(f"Page count mismatch: Base has {len(doc1)} pages vs Comparison has {len(doc2)} pages.")
        return False

    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    identical = True

    for page_num in range(len(doc1)):
        pix1 = doc1[page_num].get_pixmap(matrix=mat)
        pix2 = doc2[page_num].get_pixmap(matrix=mat)

        img1 = Image.frombytes("RGB", [pix1.width, pix1.height], pix1.samples)
        img2 = Image.frombytes("RGB", [pix2.width, pix2.height], pix2.samples)

        if img1.size != img2.size:
            st.warning(f"Page {page_num + 1}: Dimension mismatch")
            identical = False
            continue

        diff = ImageChops.difference(img1, img2)

        if diff.getbbox():
            st.write(f"Page {page_num + 1}: Differences detected.")
            identical = False

            # Highlight differences in neon green
            diff_np = np.array(diff)
            mask = np.any(diff_np > 0, axis=-1)
            
            struct = np.ones((3, 3), dtype=bool)
            mask = binary_dilation(mask, structure=struct, iterations=dilation_iterations)

            img2_np = np.array(img2)
            img2_np[mask] = [0, 255, 0]

            diff_image = Image.fromarray(img2_np)
            st.image(diff_image, caption=f"Differences on Page {page_num + 1}", use_container_width=True)
        else:
            st.write(f"Page {page_num + 1}: Identical.")

    return identical

# 2. If files are uploaded, show dropdown selectors
if uploaded_files and len(uploaded_files) >= 2:
    # Create a mapping of file names to their actual file objects
    file_dict = {file.name: file for file in uploaded_files}
    file_names = list(file_dict.keys())

    st.markdown("---")
    st.subheader("Select Files to Compare")
    
    col1, col2 = st.columns(2)
    with col1:
        base_choice = st.selectbox("Base (Reference) PDF:", file_names, index=0)
    with col2:
        # Default the second selection to index 1 if available
        comp_choice = st.selectbox("PDF to Compare against Base:", file_names, index=min(1, len(file_names)-1))

    if base_choice == comp_choice:
        st.warning("Please select two different PDF files to compare.")
    else:
        if st.button("Run Comparison"):
            file1_bytes = file_dict[base_choice].read()
            file2_bytes = file_dict[comp_choice].read()
            
            st.write(f"Comparing **{base_choice}** vs **{comp_choice}**...")
            
            result = compare_pdfs_pixel_by_pixel(file1_bytes, file2_bytes, dilation_iterations=2)
            
            if result:
                st.success(f"{base_choice} and {comp_choice} are completely identical pixel-by-pixel!")
            else:
                st.error(f"Differences found between {base_choice} and {comp_choice}.")
elif uploaded_files:
    st.info("Please upload at least 2 PDF files to use the comparison selectors.")
