"""
Script per testare il vector store senza bisogno di Groq API
"""

import chromadb
from sentence_transformers import SentenceTransformer

def test_vectorstore():
    print("🔍 Caricamento vector store...")
    
    try:
        # Carica ChromaDB
        client = chromadb.PersistentClient(path="vectorstore")
        collection = client.get_collection("appunti_corso")
        
        # Statistiche
        count = collection.count()
        print(f"✅ Vector store caricato con successo!")
        print(f"📊 Numero totale di chunks: {count}")
        
        if count == 0:
            print("⚠️ Il vector store è vuoto! Esegui prima preprocess.py")
            return
        
        # Carica modello embedding
        print("\n🧠 Caricamento modello embeddings...")
        embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("✅ Modello caricato!")
        
        # Test query
        print("\n" + "="*60)
        test_query = input("Inserisci una domanda di test (o premi ENTER per skip): ").strip()
        
        if test_query:
            print(f"\n🔍 Cerco: '{test_query}'...")
            
            query_embedding = embedding_model.encode([test_query])[0].tolist()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )
            
            print(f"\n📄 Trovati {len(results['documents'][0])} risultati rilevanti:\n")
            
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
                print(f"--- Risultato {i} ---")
                print(f"Fonte: {metadata['source']}")
                print(f"Tipo: {metadata['type']}")
                print(f"Testo: {doc[:200]}...")
                print()
        
        print("="*60)
        print("✅ Test completato! Il sistema è pronto per l'uso.")
        print("\nPer usare il chatbot completo:")
        print("1. Ottieni una Groq API key su https://console.groq.com")
        print("2. Esegui: streamlit run app.py")
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        print("\nAssicurati di aver eseguito prima: python preprocess.py")

if __name__ == "__main__":
    print("="*60)
    print("🧪 Test Vector Store")
    print("="*60)
    test_vectorstore()