"""
NER Processing Module for Sprint 2
Handles Named Entity Recognition with multiple models and EntityLinker with UMLS.

This module provides:
- Multi-model NER (Hugging Face, SciBERT, BC5CDR)
- Context detection (temporality, negation, certainty)
- EntityLinker with UMLS knowledge base
- Entity enrichment and classification
"""

import json
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Third-party imports
try:
    from transformers import pipeline
    import spacy
    import scispacy
    from scispacy.abbreviation import AbbreviationDetector
    from scispacy.linking import EntityLinker
    import numpy as np
    
    # ChromaDB integration imports
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("Please install required packages: transformers, spacy, scispacy, chromadb, sentence-transformers")


# TUI (Type Unique Identifier) Categories for UMLS classification
TUI_CATEGORIES = {
    "ENFERMEDAD": {
        "T047",  # Disease or Syndrome
        "T046",  # Pathologic Function
        "T048",  # Mental or Behavioral Dysfunction
        "T191",  # Neoplastic Process
        "T037",  # Injury or Poisoning
        "T049",  # Cell or Molecular Dysfunction
    },
    "SINTOMA": {
        "T184",  # Sign or Symptom
        "T033",  # Finding
        "T034",  # Laboratory or Test Result
    },
    "MEDICAMENTO": {
        "T121",  # Pharmacologic Substance
        "T109",  # Organic Chemical
        "T195",  # Antibiotic
        "T200",  # Clinical Drug
        "T114",  # Nucleic Acid, Nucleoside, or Nucleotide
    },
    "ANATOMIA": {
        "T029",  # Body Location or Region
        "T023",  # Body Part, Organ, or Organ Component
        "T030",  # Body Space or Junction
        "T024",  # Tissue
    },
    "PROCEDIMIENTO": {
        "T060",  # Diagnostic Procedure
        "T061",  # Therapeutic or Preventive Procedure
        "T059",  # Laboratory Procedure
    }
}


class ChromaDBLinker:
    """
    Lightweight Entity Linker using ChromaDB (avoids loading full UMLS in RAM).
    """
    def __init__(self, db_path: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize ChromaDB linker.
        
        Args:
            db_path: Path to ChromaDB persistence directory
            model_name: Sentence Transformer model name
        """
        self.db_path = db_path
        self.model_name = model_name
        self.client = None
        self.collection = None
        self.model = None
        self._load_resources()

    def _load_resources(self):
        """Load ChromaDB client and embedding model."""
        try:
            print(f"🔌 Connecting to ChromaDB at {self.db_path}...")
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.client.get_collection("umls_concepts")
            
            print(f"🧠 Loading embedding model {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            
        except Exception as e:
            raise RuntimeError(f"Error initializing ChromaDBLinker: {e}. Is the database built?")

    def link_entities(self, entities_text: List[str], top_k: int = 1) -> List[List[Dict[str, Any]]]:
        """
        Link a batch of entity texts to UMLS concepts.
        
        Args:
            entities_text: List of entity strings
            top_k: Number of results per entity
            
        Returns:
            List of result lists for each entity
        """
        if not entities_text:
            return []
            
        # Generate embeddings
        embeddings = self.model.encode(entities_text).tolist()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=embeddings,
            n_results=top_k,
            include=["metadatas", "distances", "documents"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(entities_text)):
            entity_matches = []
            
            # Check if we have results for this entity
            if not results['ids'] or len(results['ids'][i]) == 0:
                formatted_results.append([])
                continue
                
            for j in range(len(results['ids'][i])):
                match = {
                    "umls_id": results['ids'][i][j],
                    "score": 1.0 - results['distances'][i][j], # Convert distance to similarity score approx
                    "data": results['metadatas'][i][j],
                    "definition": results['documents'][i][j]
                }
                entity_matches.append(match)
            
            formatted_results.append(entity_matches)
            
        return formatted_results


class NERProcessor:
    """Processor for Named Entity Recognition with multiple models."""
    
    def __init__(self, load_advanced: bool = False, use_chromadb: bool = False, chromadb_path: str = None):
        """
        Initialize NER Processor.
        
        Args:
            load_advanced: If True, load advanced mechanisms
            use_chromadb: If True, use ChromaDB for linking instead of scispacy Linker (low RAM)
            chromadb_path: Path to ChromaDB folder
        """
        self.models_loaded = False
        self.advanced_loaded = False
        self.use_chromadb = use_chromadb
        self.chromadb_path = chromadb_path
        
        self.ner_hf = None
        self.nlp_scibert = None
        self.nlp_bc5cdr = None
        self.linker = None
        self.chroma_linker = None
        
        # Load basic models first
        self._load_basic_models()
        
        # Load advanced if requested
        if load_advanced:
            if self.use_chromadb:
                self._load_chromadb_linker()
            else:
                self._load_advanced_models()
    
    def _load_chromadb_linker(self):
        """Load ChromaDB based linker."""
        if not self.chromadb_path:
            raise ValueError("chromadb_path required for use_chromadb=True")
            
        try:
            self.chroma_linker = ChromaDBLinker(self.chromadb_path)
            self.advanced_loaded = True
            print("✅ ChromaDB Linker loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load ChromaDB Linker: {e}")
    
    def _load_basic_models(self):
        """Load basic NER models (Hugging Face, SciBERT, BC5CDR)."""
        try:
            # 1. Hugging Face NER
            self.ner_hf = pipeline(
                "token-classification",
                model="d4data/biomedical-ner-all",
                aggregation_strategy="simple"
            )
            
            # 2. SciBERT with abbreviation detector
            self.nlp_scibert = spacy.load("en_core_sci_scibert")
            self.nlp_scibert.add_pipe("abbreviation_detector")
            
            # 3. BC5CDR (specialized in diseases and chemicals)
            self.nlp_bc5cdr = spacy.load("en_ner_bc5cdr_md")
            
            self.models_loaded = True
            
        except Exception as e:
            raise RuntimeError(f"Error loading basic NER models: {e}")
    
    def _load_advanced_models(self):
        """Load advanced models (EntityLinker with UMLS)."""
        try:
            # Add EntityLinker to SciBERT pipeline
            self.nlp_scibert.add_pipe(
                "scispacy_linker",
                config={
                    "resolve_abbreviations": True,
                    "linker_name": "umls",
                    "threshold": 0.75,
                    "k": 30,
                    "max_entities_per_mention": 3
                }
            )
            
            # Get linker reference
            self.linker = self.nlp_scibert.get_pipe("scispacy_linker")
            self.advanced_loaded = True
            
        except Exception as e:
            raise RuntimeError(f"Error loading EntityLinker: {e}")
    
    @staticmethod
    def detectar_contexto(texto: str, entidad_texto: str, posicion: Tuple[int, int]) -> Dict[str, Any]:
        """
        Detect context of an entity (temporality, negation, certainty).
        
        Args:
            texto: Full text
            entidad_texto: Entity text
            posicion: Tuple (start, end) of entity
        
        Returns:
            dict with context information
        """
        inicio, fin = posicion
        
        # Get context window (50 chars before and after)
        ventana_inicio = max(0, inicio - 50)
        ventana_fin = min(len(texto), fin + 50)
        contexto = texto[ventana_inicio:ventana_fin].lower()
        
        resultado = {}
        
        # Temporal detection
        patrones_temporal = {
            "antecedente": ["antecedentes", "antecedente de", "historia de", "diagnosticado hace"],
            "actual": ["actualmente", "acude por", "presenta", "refiere", "motivo de consulta"],
            "pasado": ["hace", "desde hace", "años de evolución", "meses de evolución", "días de evolución"]
        }
        
        for tipo, patrones in patrones_temporal.items():
            if any(patron in contexto for patron in patrones):
                resultado["temporalidad"] = tipo
                break
        
        # Negation detection
        patrones_negacion = ["niega", "sin", "no presenta", "se descarta", "negativo", "ausencia de"]
        resultado["negacion"] = any(patron in contexto for patron in patrones_negacion)
        
        # Certainty detection
        if any(palabra in contexto for palabra in ["confirmado", "diagnosticado", "evidencia de"]):
            resultado["certeza"] = "confirmado"
        elif any(palabra in contexto for palabra in ["probable", "posible", "sospecha"]):
            resultado["certeza"] = "probable"
        else:
            resultado["certeza"] = "mencionado"
        
        return resultado
    
    @staticmethod
    def clasificar_por_tui(tui_codes: List[str]) -> str:
        """
        Classify entity based on TUI codes.
        
        Args:
            tui_codes: List of TUI codes
        
        Returns:
            Category name
        """
        for categoria, tuis in TUI_CATEGORIES.items():
            if any(tui in tuis for tui in tui_codes):
                return categoria
        return "OTRO"
    
    def enriquecer_entidad(self, entidad_span, linker) -> Dict[str, Any]:
        """
        Enrich entity with UMLS information.
        
        Args:
            entidad_span: Spacy span with entity
            linker: EntityLinker from scispacy
        
        Returns:
            dict with enriched information
        """
        entidad_info = {
            "texto_original": entidad_span.text,
            "tipo_ner": entidad_span.label_,
            "posicion": (entidad_span.start_char, entidad_span.end_char),
        }
        
        # Try to get UMLS information if linked
        if hasattr(entidad_span._, 'kb_ents') and entidad_span._.kb_ents:
            # Take best match (first)
            umls_id, score = entidad_span._.kb_ents[0]
            
            # Get complete UMLS information
            if umls_id in linker.kb.cui_to_entity:
                umls_entity = linker.kb.cui_to_entity[umls_id]
                
                entidad_info.update({
                    "umls_id": umls_id,
                    "nombre_normalizado": umls_entity.canonical_name,
                    "definicion": umls_entity.definition if umls_entity.definition else "Sin definición disponible",
                    "tipos_semanticos": list(umls_entity.types),
                    "categoria": self.clasificar_por_tui(umls_entity.types),
                    "sinonimos": list(umls_entity.aliases)[:5],  # Top 5 synonyms
                    "score_linking": round(float(score), 3),
                })
                
                # Get all alternatives
                alternativas = []
                for alt_id, alt_score in entidad_span._.kb_ents[1:]:
                    if alt_id in linker.kb.cui_to_entity:
                        alt_entity = linker.kb.cui_to_entity[alt_id]
                        alternativas.append({
                            "umls_id": alt_id,
                            "nombre": alt_entity.canonical_name,
                            "score": round(float(alt_score), 3)
                        })
                
                if alternativas:
                    entidad_info["alternativas"] = alternativas
        
        return entidad_info
    
    def procesar_basico(self, texto: str) -> Dict[str, Any]:
        """
        Process text with basic multi-model NER.
        
        Args:
            texto: Clinical text to process
        
        Returns:
            dict with structured results
        """
        if not self.models_loaded:
            raise RuntimeError("Basic models not loaded")
        
        # Extract entities with all models
        # 1. Hugging Face
        resultados_hf = self.ner_hf(texto)
        
        # 2. SciBERT
        doc_scibert = self.nlp_scibert(texto)
        
        # 3. BC5CDR
        doc_bc5cdr = self.nlp_bc5cdr(texto)
        
        # Structure output
        output = {
            "metadatos": {
                "fecha_procesamiento": datetime.now().isoformat(),
                "modelos_usados": [
                    "d4data/biomedical-ner-all",
                    "en_core_sci_scibert",
                    "en_ner_bc5cdr_md"
                ]
            },
            "texto_original": texto,
            "entidades_huggingface": [],
            "entidades_scibert": [],
            "entidades_bc5cdr": [],
            "abreviaturas": [],
            "estadisticas": {
                "total_hf": 0,
                "total_scibert": 0,
                "total_bc5cdr": 0
            }
        }
        
        # Process Hugging Face entities
        for ent in resultados_hf:
            entidad = {
                "texto": ent['word'],
                "tipo": ent['entity_group'],
                "score": round(float(ent['score']), 3),
                "posicion": (ent['start'], ent['end'])
            }
            output["entidades_huggingface"].append(entidad)
        output["estadisticas"]["total_hf"] = len(resultados_hf)
        
        # Process SciBERT entities
        for ent in doc_scibert.ents:
            contexto = self.detectar_contexto(texto, ent.text, (ent.start_char, ent.end_char))
            entidad = {
                "texto": ent.text,
                "tipo": ent.label_,
                "posicion": (ent.start_char, ent.end_char),
                "contexto": contexto
            }
            output["entidades_scibert"].append(entidad)
        output["estadisticas"]["total_scibert"] = len(doc_scibert.ents)
        
        # Process BC5CDR entities
        for ent in doc_bc5cdr.ents:
            contexto = self.detectar_contexto(texto, ent.text, (ent.start_char, ent.end_char))
            entidad = {
                "texto": ent.text,
                "tipo": ent.label_,
                "posicion": (ent.start_char, ent.end_char),
                "contexto": contexto
            }
            output["entidades_bc5cdr"].append(entidad)
        output["estadisticas"]["total_bc5cdr"] = len(doc_bc5cdr.ents)
        
        # Process abbreviations
        for abrv in doc_scibert._.abbreviations:
            output["abreviaturas"].append({
                "abreviatura": abrv.text,
                "forma_larga": abrv._.long_form.text,
                "posicion": (abrv.start_char, abrv.end_char)
            })
        
        return output
    
    def procesar_avanzado(self, texto: str) -> Dict[str, Any]:
        """
        Process text with advanced NER + EntityLinker (Traditional or ChromaDB).
        
        Args:
            texto: Clinical text to process
        
        Returns:
            dict with enriched structured results
        """
        if not self.advanced_loaded:
            raise RuntimeError("Advanced models not loaded. Initialize with load_advanced=True")
            
        if self.use_chromadb:
            return self._procesar_avanzado_chromadb(texto)
        
        # Traditional EntityLinker flow
        return self._procesar_avanzado_scispacy(texto)

    def _procesar_avanzado_chromadb(self, texto: str) -> Dict[str, Any]:
        """Process using ChromaDB for linking."""
        # Execute NER with SciBERT
        doc_scibert = self.nlp_scibert(texto)
        
        output = {
            "metadatos": {
                "fecha_procesamiento": datetime.now().isoformat(),
                "modelos_usados": ["en_core_sci_scibert", "all-MiniLM-L6-v2"],
                "knowledge_base": "UMLS (ChromaDB)",
                "modo": "Optimized (ChromaDB)"
            },
            "texto_original": texto,
            "entidades_por_categoria": defaultdict(list),
            "abreviaturas": [],
            "estadisticas": defaultdict(int)
        }
        
        # Process abbreviations
        for abrv in doc_scibert._.abbreviations:
            output["abreviaturas"].append({
                "abreviatura": abrv.text,
                "forma_larga": abrv._.long_form.text,
                "posicion": (abrv.start_char, abrv.end_char)
            })
            
        # Collect entities for batch linking
        entities_to_link = [ent.text for ent in doc_scibert.ents]
        
        # Batch query ChromaDB
        if entities_to_link and self.chroma_linker:
            linked_results = self.chroma_linker.link_entities(entities_to_link, top_k=3)
        else:
            linked_results = [[] for _ in entities_to_link]
            
        # Enrich and classify
        for i, ent in enumerate(doc_scibert.ents):
            matches = linked_results[i]
            
            entidad_info = {
                "texto_original": ent.text,
                "tipo_ner": ent.label_,
                "posicion": (ent.start_char, ent.end_char),
            }
            
            # Use best match
            if matches:
                best = matches[0]
                meta = best['data']
                
                # Parse types string back to list if needed
                types_list = meta['types'].split(',') if meta.get('types') else []
                
                entidad_info.update({
                    "umls_id": best['umls_id'],
                    "nombre_normalizado": meta['canonical_name'],
                    "definicion": meta.get('definition_snippet', "Sin definición"),
                    "tipos_semanticos": types_list,
                    "categoria": self.clasificar_por_tui(types_list),
                    "score_linking": round(best['score'], 3)
                })
                
                # Alternatives
                if len(matches) > 1:
                    entidad_info["alternativas"] = []
                    for alt in matches[1:]:
                        entidad_info["alternativas"].append({
                            "umls_id": alt['umls_id'],
                            "nombre": alt['data']['canonical_name'],
                            "score": round(alt['score'], 3)
                        })
            else:
                # Fallback category if no match
                entidad_info["categoria"] = "OTRO"
            
            # Add context
            contexto = self.detectar_contexto(texto, ent.text, (ent.start_char, ent.end_char))
            entidad_info["contexto"] = contexto
            
            # Add to proper category list
            cat = entidad_info.get("categoria", "OTRO")
            output["entidades_por_categoria"][cat].append(entidad_info)
            output["estadisticas"][cat] += 1
            
        # Convert defaultdicts
        output["entidades_por_categoria"] = dict(output["entidades_por_categoria"])
        output["estadisticas"] = dict(output["estadisticas"])
        
        return output

    def _procesar_avanzado_scispacy(self, texto: str) -> Dict[str, Any]:
        """
        Original logic using scispacy EntityLinker (High RAM).
        """
        # Execute NER
        doc_scibert = self.nlp_scibert(texto)
        
        # Structure output
        output = {
            "metadatos": {
                "fecha_procesamiento": datetime.now().isoformat(),
                "modelos_usados": ["en_core_sci_scibert"],
                "knowledge_base": "UMLS",
                "total_conceptos_umls": len(self.linker.kb.cui_to_entity)
            },
            "texto_original": texto,
            "entidades_por_categoria": defaultdict(list),
            "abreviaturas": [],
            "estadisticas": defaultdict(int)
        }
        
        # Process abbreviations
        for abrv in doc_scibert._.abbreviations:
            output["abreviaturas"].append({
                "abreviatura": abrv.text,
                "forma_larga": abrv._.long_form.text,
                "posicion": (abrv.start_char, abrv.end_char)
            })
        
        # Process and enrich entities
        for ent in doc_scibert.ents:
            # Enrich with UMLS
            entidad_enriquecida = self.enriquecer_entidad(ent, self.linker)
            
            # Detect context
            contexto = self.detectar_contexto(texto, ent.text, (ent.start_char, ent.end_char))
            entidad_enriquecida["contexto"] = contexto
            
            # Classify by category
            categoria = entidad_enriquecida.get("categoria", "OTRO")
            output["entidades_por_categoria"][categoria].append(entidad_enriquecida)
            output["estadisticas"][categoria] += 1
        
        # Convert defaultdict to normal dict for JSON
        output["entidades_por_categoria"] = dict(output["entidades_por_categoria"])
        output["estadisticas"] = dict(output["estadisticas"])
        
        return output


def convert_to_native_types(obj):
    """Convert numpy/tensorflow types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def export_to_json(resultados: Dict[str, Any], filename: str = "ner_resultados.json") -> str:
    """
    Export results to JSON file.
    
    Args:
        resultados: Results dictionary
        filename: Output filename
    
    Returns:
        JSON string
    """
    json_str = json.dumps(resultados, ensure_ascii=False, indent=2, default=convert_to_native_types)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(json_str)
    
    return json_str
