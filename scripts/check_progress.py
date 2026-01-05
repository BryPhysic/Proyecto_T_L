#!/usr/bin/env python3
"""Check ChromaDB indexing progress"""
import chromadb

CHROMADB_PATH = "/Users/bryphy/Proyecto_T_L/Proyecto_T_L/Datasets/chromadb_umls"

try:
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    collection = client.get_collection("umls_concepts")
    count = collection.count()
    
    total = 3920422
    percentage = (count / total) * 100
    
    print(f"📊 Progreso de Indexación UMLS")
    print(f"=" * 50)
    print(f"✅ Conceptos indexados: {count:,}")
    print(f"📦 Total conceptos:     {total:,}")
    print(f"📈 Progreso:            {percentage:.2f}%")
    print(f"=" * 50)
    
except Exception as e:
    print(f"❌ Error: {e}")
