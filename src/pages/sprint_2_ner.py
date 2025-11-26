"""
Sprint 2: NER y Estructuración
Placeholder page for Named Entity Recognition and clinical text structuring.
"""

import streamlit as st


def render(translations: dict, lang: str):
    """
    Render the Sprint 2 placeholder page.
    
    Args:
        translations: Dictionary with UI translations
        lang: Current language code ('es' or 'en')
    """
    st.title("🔖 Sprint 2: NER y Estructuración" if lang == 'es' else "🔖 Sprint 2: NER and Structuring")
    
    st.info("⏳ " + translations['coming_soon'])
    
    st.markdown("---")
    
    # Description
    if lang == 'es':
        st.markdown("""
        ### Objetivo
        
        Extraer entidades clínicas de notas médicas y convertirlas en datos estructurados (JSON).
        
        ### Funcionalidades Planeadas
        
        - 🏥 Extracción de entidades biomédicas (síntomas, medicamentos, diagnósticos)
        - 📊 Visualización de entidades extraídas
        - 💾 Exportación a formato JSON estructurado
        - 📈 Métricas de confianza por entidad
        
        ### Modelo
        
        Se utilizará un modelo de NER biomédico pre-entrenado para identificar:
        - Enfermedades y condiciones
        - Medicamentos y tratamientos
        - Síntomas y signos vitales
        - Procedimientos médicos
        """)
    else:
        st.markdown("""
        ### Objective
        
        Extract clinical entities from medical notes and convert them into structured data (JSON).
        
        ### Planned Features
        
        - 🏥 Biomedical entity extraction (symptoms, medications, diagnoses)
        - 📊 Visualization of extracted entities
        - 💾 Export to structured JSON format
        - 📈 Confidence metrics per entity
        
        ### Model
        
        A pre-trained biomedical NER model will be used to identify:
        - Diseases and conditions
        - Medications and treatments
        - Symptoms and vital signs
        - Medical procedures
        """)
    
    st.markdown("---")
    
    # Link to notebook
    st.markdown(
        "📓 " + translations['see_notebook'] + ": `notebooks/2_ner_estructurador.ipynb`"
    )
