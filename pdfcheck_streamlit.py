import streamlit as st
import fitz  # PyMuPDF
from PIL import Image, ImageChops
import numpy as np
from scipy.ndimage import binary_dilation

# Expand Streamlit to use the full width of the screen
st.set_page_config(page_title="PDF Comparison Tool", layout="wide")

st.title("PDF Pixel-by-Pixel Comparison")

# 1. Upload multiple files at once with professional labeling
uploaded_files = st.file_uploader(
    "Upload your PDF files (multiple files accepted; select two to compare against each other):", 
    type="pdf", 
    accept_multiple_files=True
)

def compare_pdfs_pixel_by_pixel(pdf_bytes_1, pdf_bytes_2, dpi=300, dilation_iterations=2):
    doc1 = fitz.open(stream=pdf_bytes_1, filetype="pdf")
    doc2 = fitz.open(stream=pdf_bytes_2, filetype="pdf")

    len1, len2 = len(doc1), len(doc2)
    if len1 != len2:
        st.warning(f"Page count mismatch: Base has {len1} pages, but Comparison has {len2} pages. Comparing overlapping pages...")

    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    identical = True
    
    # Compare up to the page count of the shorter document
    min_pages = min(len1, len2)

    for page_num in range(min_pages):
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

    # Flag if one document has extra pages
    if len1 != len2:
        identical = False
        if len2 > len1:
            st.info(f"The comparison PDF has {len2 - len1} extra page(s) at the end (Pages {len1 + 1} to {len2}).")
        else:
            st.info(f"The base PDF has {len1 - len2} extra page(s) at the end (Pages {len2 + 1} to {len1}).")

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
                st.success(f"PDF comparison completed between **{base_choice}** and **{comp_choice}**: They are completely identical pixel-by-pixel!")
            else:
                st.error(f"PDF comparison completed between **{base_choice}** and **{comp_choice}**: Differences found and highlighted above.")
elif uploaded_files:
    st.info("Please upload at least 2 PDF files to use the comparison selectors.")

# Footer
st.markdown("---")
st.caption("(c) Anh nguyen 2026")
