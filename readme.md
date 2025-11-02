# 🎓 RAG Chatbot per Appunti Universitari

Chatbot intelligente che risponde a domande sui tuoi appunti usando RAG (Retrieval-Augmented Generation).

## 🌟 Caratteristiche

- ✅ **100% Gratuito** - usa Groq API (gratis)
- 📄 **PDF & Immagini** - supporta entrambi i formati
- 🧠 **RAG avanzato** - cerca negli appunti e genera risposte contestuali
- ☁️ **Deploy facile** - pronto per Streamlit Cloud
- 🇮🇹 **Multilingua** - ottimizzato per italiano

## 📋 Prerequisiti

### Per Pre-processing Locale
1. Python 3.9+
2. Tesseract OCR installato:
   - **Windows**: [Download qui](https://github.com/UB-Mannheim/tesseract/wiki)
   - **Mac**: `brew install tesseract tesseract-lang`
   - **Linux**: `sudo apt-get install tesseract-ocr tesseract-ocr-ita`

### Per Deploy su Streamlit Cloud
1. Account GitHub (gratuito)
2. Account Streamlit Cloud (gratuito)
3. Groq API Key (gratuita) - [Registrati qui](https://console.groq.com)

## 🚀 Installazione e Setup

### Passo 1: Setup Locale

```bash
# Clona o crea il progetto
mkdir rag-chatbot
cd rag-chatbot

# Installa dipendenze per pre-processing
pip install -r requirements-local.txt

# Crea struttura cartelle
mkdir -p data/appunti
```

### Passo 2: Aggiungi i tuoi Appunti

Metti tutti i tuoi PDF e immagini nella cartella `data/appunti/`:

```
data/
└── appunti/
    ├── lezione1.pdf
    ├── lezione2.pdf
    ├── schema1.jpg
    ├── schema2.png
    └── ...
```

### Passo 3: Pre-processa i Documenti

```bash
# Esegui lo script di pre-processing
python preprocess.py
```

Questo script:
- 📖 Legge PDF e immagini
- 🔤 Estrae testo (OCR per immagini)
- ✂️ Divide in chunks
- 🧮 Crea embeddings
- 💾 Salva nel vector database (`vectorstore/`)

**⏱️ Tempo**: dipende dal numero di documenti (es. 10-20 min per 100 pagine)

### Passo 4: Test Locale (Opzionale)

```bash
# Installa streamlit
pip install -r requirements.txt

# Testa l'app localmente
streamlit run app.py
```

## ☁️ Deploy su Streamlit Cloud

### Passo 1: Prepara Repository GitHub

```bash
# Inizializza git (se non già fatto)
git init

# Aggiungi tutti i file (INCLUSO vectorstore!)
git add .
git commit -m "Initial commit"

# Crea repo su GitHub e pusha
git remote add origin https://github.com/TUO_USERNAME/rag-chatbot.git
git push -u origin main
```

**⚠️ IMPORTANTE**: Assicurati che la cartella `vectorstore/` sia nel repository!

### Passo 2: Deploy su Streamlit

1. Vai su [share.streamlit.io](https://share.streamlit.io)
2. Clicca "New app"
3. Seleziona il tuo repository GitHub
4. Main file: `app.py`
5. Clicca "Deploy"

### Passo 3: Ottieni Groq API Key

1. Vai su [console.groq.com](https://console.groq.com)
2. Registrati (gratis)
3. Crea una nuova API Key
4. Copiala e usala nell'app Streamlit

## 🎯 Come Usare il Chatbot

1. **Apri l'app** (locale o su Streamlit Cloud)
2. **Inserisci la Groq API Key** nella sidebar
3. **Fai domande** sui tuoi appunti:
   - "Spiega il concetto di X"
   - "Quali sono le formule per Y?"
   - "Riassumi la lezione su Z"
4. **Visualizza le fonti** cliccando sull'expander sotto ogni risposta

## 📁 Struttura Progetto

```
rag-chatbot/
├── app.py                      # App Streamlit principale
├── preprocess.py               # Script pre-processing locale
├── requirements.txt            # Dipendenze per Streamlit Cloud
├── requirements-local.txt      # Dipendenze per pre-processing
├── README.md                   # Questa guida
├── data/
│   └── appunti/               # I tuoi PDF e immagini (locale)
└── vectorstore/               # Database vettoriale (da committare su GitHub)
    └── chroma.sqlite3         # ChromaDB
```

## 🔧 Configurazioni Avanzate

### Modificare Chunk Size

In `preprocess.py`, linea ~66:

```python
chunks = self.chunk_text(text, chunk_size=500, overlap=50)
```

- **chunk_size**: dimensione chunk (più grande = più contesto, meno precisione)
- **overlap**: sovrapposizione tra chunks (evita di tagliare frasi)

### Cambiare Numero di Risultati

In `app.py`, linea ~43:

```python
results = collection.query(..., n_results=3)
```

- Aumenta per avere più contesto (ma risposte più lunghe)
- Diminuisci per risposte più mirate

### Usare Modelli Diversi

**Groq** offre vari modelli gratuiti. In `app.py`, linea ~64:

```python
model="llama-3.1-70b-versatile"  # Più intelligente
# model="llama-3.1-8b-instant"   # Più veloce
# model="mixtral-8x7b-32768"     # Più contesto
```

## ⚠️ Limitazioni

- **Streamlit Cloud**: 1GB RAM - se hai troppi appunti, il vector store potrebbe essere troppo grande
- **Groq API**: Rate limits gratuiti (30 req/min) - sufficiente per uso personale
- **OCR**: La qualità dipende dalla qualità delle immagini

## 🆘 Troubleshooting

### "Collection not found"
- Assicurati di aver eseguito `preprocess.py`
- Verifica che `vectorstore/` sia nel repository

### "Tesseract not found"
- Installa Tesseract OCR (vedi prerequisiti)
- Su Windows, aggiungi al PATH: `C:\Program Files\Tesseract-OCR`

### "Out of memory" su Streamlit Cloud
- Riduci il numero di documenti
- Aumenta chunk_size per avere meno chunks totali

### Risposte non pertinenti
- Aumenta `n_results` per più contesto
- Verifica che l'OCR abbia estratto correttamente il testo
- Controlla i chunks in `vectorstore/` aprendo ChromaDB

## 📚 Risorse

- [Groq Documentation](https://console.groq.com/docs)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Sentence Transformers](https://www.sbert.net/)

## 🎓 Suggerimenti per l'Uso

1. **Organizza gli appunti**: Usa nomi file descrittivi (es. `capitolo3-termodinamica.pdf`)
2. **Qualità immagini**: Per OCR migliore, usa immagini ad alta risoluzione
3. **Domande specifiche**: Funziona meglio con domande precise piuttosto che generiche
4. **Controlla le fonti**: Verifica sempre quali documenti ha usato per rispondere

## 📝 Licenza

MIT License - Libero per uso personale e accademico

---

**Fatto con ❤️ per studenti universitari**

Per domande o problemi, apri una issue su GitHub!