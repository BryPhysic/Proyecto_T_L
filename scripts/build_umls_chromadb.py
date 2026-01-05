import json
import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import sys
from tqdm import tqdm

# Configuration
UMLS_FILE_PATH = "/Users/bryphy/Proyecto_T_L/Proyecto_T_L/Datasets/datasets/d5e593bc2d8adeee7754be423cd64f5d331ebf26272074a2575616be55697632.0660f30a60ad00fffd8bbf084a18eb3f462fd192ac5563bf50940fc32a850a3c.umls_2022_ab_cat0129.jsonl"
CHROMADB_PATH = "/Users/bryphy/Proyecto_T_L/Proyecto_T_L/Datasets/chromadb_umls"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 1000  # Conservative batch size for 4-8GB RAM
MAX_CONCEPTS = None # Set to None to process all, or int for testing (e.g., 50000)

def build_vector_db():
    print(f"🚀 Starting UMLS indexing to ChromaDB...")
    print(f"📂 Input: {UMLS_FILE_PATH}")
    print(f"💾 Output: {CHROMADB_PATH}")
    print(f"🧠 Model: {MODEL_NAME}")
    
    # Check if input file exists
    if not os.path.exists(UMLS_FILE_PATH):
        print(f"❌ Error: Input file not found at {UMLS_FILE_PATH}")
        return

    # Initialize ChromaDB
    # client = chromadb.PersistentClient(path=CHROMADB_PATH)
    # Using new client initialization for newer versions if needed, but standard is PersistentClient
    try:
        client = chromadb.PersistentClient(path=CHROMADB_PATH)
    except Exception as e:
        print(f"❌ Error initializing ChromaDB: {e}")
        return

    # Create or get collection
    try:
        collection = client.get_or_create_collection(
            name="umls_concepts",
            metadata={"hnsw:space": "cosine"} # Cosine distance for text similarity
        )
        print(f"✅ Collection 'umls_concepts' ready.")
    except Exception as e:
        print(f"❌ Error creating collection: {e}")
        return

    # Initialize Embedding Model
    print("⏳ Loading embedding model...")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Try installing: pip install sentence-transformers")
        return

    # Count total lines for progress bar (optional, takes time but good for UX)
    # print("⏳ Counting lines...")
    # total_lines = sum(1 for _ in open(UMLS_FILE_PATH))
    # print(f"📊 Total concepts to process: {total_lines}")
    
    # Processing Loop
    documents = []
    metadatas = []
    ids = []
    
    count = 0
    batch_count = 0
    
    print("🚀 Indexing concepts...")
    
    with open(UMLS_FILE_PATH, 'r') as f:
        for line in tqdm(f, desc="Processing UMLS"):
            try:
                data = json.loads(line)
                
                # Extract fields
                concept_id = data.get("concept_id")
                canonical_name = data.get("canonical_name", "")
                definition = data.get("definition", "")
                types = data.get("types", [])
                aliases = data.get("aliases", [])
                
                if not concept_id or not canonical_name:
                    continue
                
                # Construct RAG-friendly document content
                # This is what RAG will "read" to generate answers
                doc_content = f"Concept: {canonical_name}\n"
                if definition:
                    doc_content += f"Definition: {definition}\n"
                
                # Prepare metadata for filtering and frontend display
                meta = {
                    "cui": concept_id,
                    "canonical_name": canonical_name,
                    "types": ",".join(types) if types else "",
                    "has_definition": bool(definition),
                    "definition_snippet": definition[:100] + "..." if len(definition) > 100 else definition
                }
                
                # Add to batch
                documents.append(doc_content)
                metadatas.append(meta)
                ids.append(concept_id)
                
                count += 1
                
                # Flush batch
                if len(documents) >= BATCH_SIZE:
                    # Generate embeddings
                    embeddings = model.encode(documents).tolist()
                    
                    # Add to ChromaDB
                    collection.add(
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                    
                    # Reset batch
                    documents = []
                    metadatas = []
                    ids = []
                    batch_count += 1
                    
                    if batch_count % 10 == 0:
                        print(f"   Processed {count} concepts...")
                        
                if MAX_CONCEPTS and count >= MAX_CONCEPTS:
                    print(f"⚠️ Limit of {MAX_CONCEPTS} reached. Stopping.")
                    break
                    
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"⚠️ Error processing line: {e}")
                continue

    # Flush remaining
    if documents:
        print("⏳ Flushing final batch...")
        embeddings = model.encode(documents).tolist()
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    print("\n" + "="*50)
    print(f"🎉 DONE! Successfully indexed {count} concepts.")
    print(f"💾 Database saved to: {CHROMADB_PATH}")
    print("="*50)

if __name__ == "__main__":
    build_vector_db()
