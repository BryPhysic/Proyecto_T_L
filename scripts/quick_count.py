#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/bryphy/Proyecto_T_L/Proyecto_T_L/.venv/lib/python3.10/site-packages')

import chromadb

try:
    client = chromadb.PersistentClient(path="/Users/bryphy/Proyecto_T_L/Proyecto_T_L/Datasets/chromadb_umls")
    collection = client.get_collection("umls_concepts")
    count = collection.count()
    print(count)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
