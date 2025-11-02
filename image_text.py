from pdf2image import convert_from_path
import pytesseract
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# Convert PDF pages to images
pages = convert_from_path('data/imagepdf/Asset Pricing Notes.pdf')

# Create a new PDF with extracted text
output_pdf = canvas.Canvas('data/appunti/extracted_text.pdf', pagesize=A4)
width, height = A4

# Extract text from each page and add to PDF
for i, page in enumerate(pages):
    text = pytesseract.image_to_string(page)
    
    # Start a new page for each image
    if i > 0:
        output_pdf.showPage()
    
    # Add text to PDF (simple version)
    text_object = output_pdf.beginText(40, height - 40)
    text_object.setFont("Helvetica", 10)
    
    # Split text into lines and add them
    for line in text.split('\n'):
        text_object.textLine(line)
    
    output_pdf.drawText(text_object)

# Save the PDF
output_pdf.save()