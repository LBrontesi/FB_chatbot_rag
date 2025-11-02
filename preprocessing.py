"""
Script per pre-processare PDF e immagini localmente.
Esegui questo script sul tuo computer prima di deployare su Streamlit Cloud.

Requisiti:
- pip install pypdf2 pillow pytesseract chromadb sentence-transformers
- Installare Tesseract OCR: https://github.com/tesseract-ocr/tesseract
"""

import os
from pathlib import Path
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import uuid

class DocumentProcessor:
    def __init__(self, data_folder: str = "data/appunti", vectorstore_folder: str = "vectorstore"):
        self.data_folder = Path(data_folder)
        self.vectorstore_folder = Path(vectorstore_folder)
        self.vectorstore_folder.mkdir(exist_ok=True)
        
        # Inizializza embedding model
        print("Caricamento modello embeddings...")
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Inizializza ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.vectorstore_folder))
        
        # Crea o ottieni collection
        try:
            self.collection = self.client.get_collection("appunti_corso")
            print("Collection esistente trovata. Eliminazione in corso...")
            self.client.delete_collection("appunti_corso")
        except:
            pass
        
        self.collection = self.client.create_collection(
            name="appunti_corso",
            metadata={"hnsw:space": "cosine"}
        )
        print("Nuova collection creata!")
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Estrae testo da PDF"""
        try:
            reader = PdfReader(str(pdf_path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            print(f"Errore lettura PDF {pdf_path.name}: {e}")
            return ""
    
    def extract_text_from_image(self, image_path: Path) -> str:
        """Estrae testo da immagine usando OCR"""
        try:
            image = Image.open(image_path)
            # OCR con tesseract (supporta italiano)
            text = pytesseract.image_to_string(image, lang='ita+eng')
            return text.strip()
        except Exception as e:
            print(f"Errore OCR immagine {image_path.name}: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Divide il testo in chunks con overlap"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def process_documents(self):
        """Processa tutti i documenti nella cartella data"""
        if not self.data_folder.exists():
            print(f"Cartella {self.data_folder} non trovata!")
            return
        
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        # Supporta PDF e immagini comuni
        supported_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        files = [f for f in self.data_folder.rglob('*') if f.suffix.lower() in supported_extensions]
        
        print(f"\nTrovati {len(files)} file da processare\n")
        
        for file_path in files:
            print(f"Processing: {file_path.name}...")
            
            # Estrai testo
            if file_path.suffix.lower() == '.pdf':
                text = self.extract_text_from_pdf(file_path)
            else:
                text = self.extract_text_from_image(file_path)
            
            if not text:
                print(f"  ⚠️  Nessun testo estratto da {file_path.name}")
                continue
            
            # Dividi in chunks
            chunks = self.chunk_text(text)
            print(f"  ✓ Estratti {len(chunks)} chunks")
            
            # Prepara per ChromaDB
            for idx, chunk in enumerate(chunks):
                all_documents.append(chunk)
                all_metadatas.append({
                    "source": file_path.name,
                    "chunk_id": idx,
                    "type": "pdf" if file_path.suffix.lower() == '.pdf' else "image"
                })
                all_ids.append(str(uuid.uuid4()))
        
        # Aggiungi tutto a ChromaDB
        if all_documents:
            print(f"\n📊 Creazione embeddings per {len(all_documents)} chunks...")
            
            # ChromaDB può gestire gli embeddings automaticamente o possiamo fornirli
            # Qui li calcoliamo noi per avere più controllo
            embeddings = self.embedding_model.encode(all_documents, show_progress_bar=True)
            
            print("💾 Salvataggio nel vector store...")
            self.collection.add(
                documents=all_documents,
                embeddings=embeddings.tolist(),
                metadatas=all_metadatas,
                ids=all_ids
            )
            
            print(f"\n✅ COMPLETATO! {len(all_documents)} chunks salvati in {self.vectorstore_folder}")
            print(f"📁 Ora puoi caricare la cartella '{self.vectorstore_folder}' su GitHub")
        else:
            print("\n⚠️  Nessun documento processato!")

def main():
    print("="*60)
    print("🚀 RAG Chatbot - Pre-processing Documenti")
    print("="*60)
    
    processor = DocumentProcessor()
    processor.process_documents()
    
    print("\n" + "="*60)
    print("📝 Prossimi passi:")
    print("1. Verifica che la cartella 'vectorstore' sia stata creata")
    print("2. Carica tutto il progetto (incluso vectorstore) su GitHub")
    print("3. Deploya su Streamlit Cloud")
    print("="*60)

if __name__ == "__main__":
    main()