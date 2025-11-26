"""
Sprint 4: RAG sobre Guías Clínicas
Placeholder page for Retrieval-Augmented Generation over clinical guidelines.
"""

import streamlit as st


def render(translations: dict, lang: str):
    """
    Render the Sprint 4 placeholder page.
    
    Args:
        translations: Dictionary with UI translations
        lang: Current language code ('es' or 'en')
    """
    st.title("💬 Sprint 4: RAG sobre Guías Clínicas" if lang == 'es' 
             else "💬 Sprint 4: RAG on Clinical Guidelines")
    
    st.info("⏳ " + translations['coming_soon'])
    
    st.markdown("---")
    
    # Description
    if lang == 'es':
        st.markdown("""
        ### Objetivo
        
        Construir un sistema de consulta que recupere información de guías clínicas y responda con evidencia citada.
        
        ### Funcionalidades Planeadas
        
        - 🔍 Búsqueda semántica sobre guías y protocolos clínicos
        - 📚 Base de conocimiento con documentación médica confiable
        - 💬 Chat interactivo con referencias bibliográficas
        - 🎯 Respuestas con citas y fuentes verificables
        
        ### Arquitectura RAG
        
        1. **Indexación**: Vectorización de guías clínicas usando embeddings
        2. **Recuperación**: Búsqueda semántica de pasajes relevantes
        3. **Generación**: LLM genera respuesta basada en contexto recuperado
        4. **Citación**: Inclusión de referencias a las fuentes originales
        
        ### Consideraciones Éticas
        
        - ⚠️ Las respuestas deben siempre indicar que no reemplazan consulta médica
        - 📖 Todas las afirmaciones deben estar respaldadas por fuentes citadas
        - 🔒 Información sensible debe manejarse con privacidad
        """)
    else:
        st.markdown("""
        ### Objective
        
        Build a query system that retrieves information from clinical guidelines and responds with cited evidence.
        
        ### Planned Features
        
        - 🔍 Semantic search over clinical guidelines and protocols
        - 📚 Knowledge base with reliable medical documentation
        - 💬 Interactive chat with bibliographic references
        - 🎯 Answers with citations and verifiable sources
        
        ### RAG Architecture
        
        1. **Indexing**: Vectorization of clinical guidelines using embeddings
        2. **Retrieval**: Semantic search for relevant passages
        3. **Generation**: LLM generates response based on retrieved context
        4. **Citation**: Inclusion of references to original sources
        
        ### Ethical Considerations
        
        - ⚠️ Responses must always indicate they don't replace medical consultation
        - 📖 All claims must be backed by cited sources
        - 🔒 Sensitive information must be handled with privacy
        """)
    
    st.markdown("---")
    
    # Link to notebook
    st.markdown(
        "📓 " + translations['see_notebook'] + ": `notebooks/4_rag_chat.ipynb`"
    )
