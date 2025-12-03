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
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("Please install required packages: transformers, spacy, scispacy")


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


class NERProcessor:
    """Processor for Named Entity Recognition with multiple models."""
    
    def __init__(self, load_advanced: bool = False):
        """
        Initialize NER Processor.
        
        Args:
            load_advanced: If True, load EntityLinker with UMLS (slower, ~1GB download first time)
        """
        self.models_loaded = False
        self.advanced_loaded = False
        self.ner_hf = None
        self.nlp_scibert = None
        self.nlp_bc5cdr = None
        self.linker = None
        
        # Load basic models first
        self._load_basic_models()
        
        # Load advanced if requested
        if load_advanced:
            self._load_advanced_models()
    
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
        Process text with advanced NER + EntityLinker + UMLS.
        
        Args:
            texto: Clinical text to process
        
        Returns:
            dict with enriched structured results
        """
        if not self.advanced_loaded:
            raise RuntimeError("Advanced models not loaded. Initialize with load_advanced=True")
        
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
