from google.colab import files
import fitz  # PyMuPDF
from PIL import Image, ImageChops
import numpy as np
from scipy.ndimage import binary_dilation
from IPython.display import display

print("Please upload your PDF files (you can select multiple files at once):")
uploaded = files.upload()

pdf_files = sorted(list(uploaded.keys()))

if len(pdf_files) < 2:
    print("Error: Please upload at least 2 PDF files to compare.")
else:
    def compare_pdfs_pixel_by_pixel(pdf_path_1, pdf_path_2, output_diff_path="diff_result", dpi=150, dilation_iterations=2):
        doc1 = fitz.open(pdf_path_1)
        doc2 = fitz.open(pdf_path_2)

        if len(doc1) != len(doc2):
            print(f"Page count mismatch: {len(doc1)} vs {len(doc2)}")
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
                print(f"Page {page_num + 1}: Dimension mismatch")
                identical = False
                continue

            diff = ImageChops.difference(img1, img2)

            if diff.getbbox():
                print(f"Page {page_num + 1}: Differences detected.")
                identical = False

                # Highlight differences in pink and make them larger/thicker
                diff_np = np.array(diff)
                mask = np.any(diff_np > 0, axis=-1)

                # Expand the mask to make highlights larger (increase 'iterations' for a thicker mark)
                struct = np.ones((3, 3), dtype=bool)
                mask = binary_dilation(mask, structure=struct, iterations=dilation_iterations)

                img2_np = np.array(img2)
                #img2_np[mask] = [255, 105, 180]  # Pink color
                img2_np[mask] = [0, 255, 0]  # Green color

                diff_image = Image.fromarray(img2_np)
                out_filename = f"{output_diff_path}_page_{page_num + 1}.png"
                diff_image.save(out_filename)

                print(f"-> Displaying diff for page {page_num + 1}:")
                display(diff_image)
            else:
                print(f"Page {page_num + 1}: Identical.")

        return identical

    # Loop through pairs of uploaded files
    for i in range(0, len(pdf_files) - 1, 2):
        file1 = pdf_files[i]
        file2 = pdf_files[i+1]

        print(f"\n========================================")
        print(f"Comparing: {file1} vs {file2}")
        print(f"========================================")

        # You can change dilation_iterations=2 to 3 or 4 if you want them even thicker
        result = compare_pdfs_pixel_by_pixel(file1, file2, output_diff_path=f"diff_{file1}_vs_{file2}", dilation_iterations=2)

        if result:
            print(f"\n{file1} and {file2} are completely identical pixel-by-pixel!")
        else:
            print(f"\nDifferences found between {file1} and {file2}. Highlighted images saved.")

