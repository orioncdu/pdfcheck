import streamlit as st
import fitz  # PyMuPDF
from PIL import Image, ImageChops
import numpy as np
from scipy.ndimage import binary_dilation

st.title("PDF Pixel-by-Pixel Comparison Tool")

# Streamlit multi-file uploader
uploaded_files = st.file_uploader(
    "Upload your PDF files (select multiple at once):", 
    type="pdf", 
    accept_multiple_files=True
)

def compare_pdfs_pixel_by_pixel(pdf_bytes_1, pdf_bytes_2, dpi=150, dilation_iterations=2):
    # Open PDFs directly from bytes using PyMuPDF
    doc1 = fitz.open(stream=pdf_bytes_1, filetype="pdf")
    doc2 = fitz.open(stream=pdf_bytes_2, filetype="pdf")

    if len(doc1) != len(doc2):
        st.error(f"Page count mismatch: {len(doc1)} vs {len(doc2)}")
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

            # Highlight differences in bright green or yellow (great for contrast)
            diff_np = np.array(diff)
            mask = np.any(diff_np > 0, axis=-1)
            
            struct = np.ones((3, 3), dtype=bool)
            mask = binary_dilation(mask, structure=struct, iterations=dilation_iterations)

            img2_np = np.array(img2)
            img2_np[mask] = [0, 255, 0]  # Neon Green highlight

            diff_image = Image.fromarray(img2_np)
            st.image(diff_image, caption=f"Differences on Page {page_num + 1}", use_container_width=True)
        else:
            st.write(f"Page {page_num + 1}: Identical.")

    return identical

# Process files when uploaded in pairs
if uploaded_files and len(uploaded_files) >= 2:
    for i in range(0, len(uploaded_files) - 1, 2):
        file1 = uploaded_files[i]
        file2 = uploaded_files[i+1]
        
        st.subheader(f"Comparing: {file1.name} vs {file2.name}")
        
        # Read file contents as bytes
        result = compare_pdfs_pixel_by_pixel(file1.read(), file2.read(), dilation_iterations=2)
        
        if result:
            st.success(f"{file1.name} and {file2.name} are completely identical pixel-by-pixel!")
        else:
            st.error(f"Differences found between {file1.name} and {file2.name}.")
elif uploaded_files:
    st.info("Please upload at least 2 PDF files to begin comparison.")
