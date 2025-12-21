import os
from PyPDF2 import PdfWriter, PdfReader

input_pdf = "App/PhD_certificate.pdf"
output_pdf = "App/phd_diploma.pdf"

original_size = os.path.getsize(input_pdf)

pdf_in = PdfReader(open(input_pdf, 'rb'))
pdf_out = PdfWriter()

for page in range(len(pdf_in.pages)):
    pdf_out.add_page(pdf_in.pages[page].compress_content_streams())

with open(output_pdf, 'wb') as f:
    pdf_out.write(f, optimized_size) 

optimized_size = os.path.getsize(output_pdf)

print(f"Original Size: {original_size}")
print(f"Optimized Size: {optimized_size}")
print(f"Reduction: {round((original_size-optimized_size)/original_size*100, 2)}%")