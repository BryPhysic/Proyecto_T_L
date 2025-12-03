"""
Asistente Clínico Inteligente y Explicable (ACIE)
==================================================
Streamlit application for clinical assistant with multiple sprints.

Main entry point with bilingual support and sprint navigation.
"""

import streamlit as st
from pages import sprint_1_triaje, sprint_2_ner, sprint_3_soap, sprint_4_rag


# Page configuration
st.set_page_config(
    page_title="ACIE - Asistente Clínico Inteligente",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Translations dictionary
TRANSLATIONS = {
    'es': {
        # General
        'app_title': 'Asistente Clínico Inteligente y Explicable',
        'app_subtitle': 'Sistema de IA para apoyo clínico - UPCH',
        'select_language': 'Seleccionar Idioma',
        'project_info': 'Información del Proyecto',
        'sprint_navigation': 'Navegación por Sprints',
        'project_progress': 'Progreso del Proyecto',
        'completed': 'Completado',
        'in_progress': 'En Progreso',
        'pending': 'Pendiente',
        
        # Sprint titles
        'sprint1_title': 'Triaje Zero-Shot',
        'sprint2_title': 'NER y Estructuración',
        'sprint3_title': 'Generador SOAP',
        'sprint4_title': 'RAG Guías Clínicas',
        
        # Sprint 1
        'configuration': 'Configuración',
        'use_custom_labels': 'Usar etiquetas personalizadas',
        'custom_labels_help': 'Activar para definir tus propias categorías de triaje',
        'enter_labels': 'Ingresa las etiquetas (separadas por comas)',
        'labels_placeholder': 'Ej: Urgencia Alta, Urgencia Media, Urgencia Baja, No Urgente',
        'labels_help': 'Define categorías claras y diferenciadas',
        'no_labels_warning': '⚠️ Debes ingresar al menos una etiqueta',
        'default_labels': 'Etiquetas predefinidas',
        'confidence_threshold': 'Umbral de Confianza',
        'threshold_help': 'Clasificaciones por debajo de este umbral se marcarán como de baja confianza',
        'current_threshold': 'Umbral actual',
        
        # Input
        'input_messages': 'Ingreso de Mensajes',
        'single_message': 'Mensaje Individual',
        'multiple_messages': 'Múltiples Mensajes',
        'examples': 'Ejemplos',
        'enter_message': 'Escribe el mensaje del paciente',
        'message_placeholder': 'Ej: Tengo un dolor fuerte en el pecho...',
        'classify_button': '🔍 Clasificar',
        'empty_message_warning': '⚠️ Por favor ingresa al menos un mensaje',
        'enter_multiple': 'Ingresa múltiples mensajes (uno por línea)',
        'multiple_placeholder': 'Mensaje 1\nMensaje 2\nMensaje 3...',
        'multiple_help': 'Cada línea será clasificada como un mensaje separado',
        'examples_info': 'Selecciona uno o más mensajes de ejemplo para clasificar:',
        'classify_selected': '🔍 Clasificar Seleccionados',
        'no_examples_warning': '⚠️ Selecciona al menos un ejemplo',
        
        # Results
        'results': 'Resultados',
        'classifying': '⏳ Clasificando mensajes...',
        'message': 'Mensaje',
        'predicted_category': 'Categoría Predicha',
        'confidence': 'Confianza',
        'low_confidence': 'Baja confianza',
        'high_confidence': 'Alta',
        'acceptable': 'Aceptable',
        'low': 'Baja',
        'no_prediction': '❌ No se pudo generar predicción',
        'probability_distribution': 'Distribución de Probabilidades',
        'category': 'Categoría',
        'threshold': 'Umbral',
        
        # Export
        'export_results': 'Exportar Resultados',
        'download_csv': 'Descargar CSV',
        'download_json': 'Descargar JSON',
        
        # Documentation
        'what_is_this': '¿Qué es esto?',
        'ethics_title': 'Consideraciones Éticas',
        
        # Sprint 2 NER
        'abbreviations': 'Abreviaturas',
        'results': 'Resultados',
        
        # Placeholders
        'coming_soon': 'Próximamente - Esta funcionalidad está en desarrollo',
        'see_notebook': 'Ver notebook',
        
        # About
        'about_text': '''
        Este proyecto docente–experimental permite construir un **asistente clínico inteligente y explicable** 
        usando modelos tipo Transformer y LLM, con énfasis en IA responsable en salud.
        
        **Sprints:**
        1. Triaje de mensajes (Zero-Shot)
        2. Estructuración de texto clínico (NER → JSON)
        3. Generación de notas SOAP con auto-auditoría
        4. Recuperación aumentada (RAG) sobre guías clínicas
        ''',
    },
    'en': {
        # General
        'app_title': 'Intelligent and Explainable Clinical Assistant',
        'app_subtitle': 'AI System for Clinical Support - UPCH',
        'select_language': 'Select Language',
        'project_info': 'Project Information',
        'sprint_navigation': 'Sprint Navigation',
        'project_progress': 'Project Progress',
        'completed': 'Completed',
        'in_progress': 'In Progress',
        'pending': 'Pending',
        
        # Sprint titles
        'sprint1_title': 'Zero-Shot Triage',
        'sprint2_title': 'NER and Structuring',
        'sprint3_title': 'SOAP Generator',
        'sprint4_title': 'RAG Clinical Guidelines',
        
        # Sprint 1
        'configuration': 'Configuration',
        'use_custom_labels': 'Use custom labels',
        'custom_labels_help': 'Enable to define your own triage categories',
        'enter_labels': 'Enter labels (comma-separated)',
        'labels_placeholder': 'E.g: High Urgency, Medium Urgency, Low Urgency, Non-Urgent',
        'labels_help': 'Define clear and differentiated categories',
        'no_labels_warning': '⚠️ You must enter at least one label',
        'default_labels': 'Predefined labels',
        'confidence_threshold': 'Confidence Threshold',
        'threshold_help': 'Classifications below this threshold will be marked as low confidence',
        'current_threshold': 'Current threshold',
        
        # Input
        'input_messages': 'Message Input',
        'single_message': 'Single Message',
        'multiple_messages': 'Multiple Messages',
        'examples': 'Examples',
        'enter_message': 'Enter patient message',
        'message_placeholder': 'E.g: I have severe chest pain...',
        'classify_button': '🔍 Classify',
        'empty_message_warning': '⚠️ Please enter at least one message',
        'enter_multiple': 'Enter multiple messages (one per line)',
        'multiple_placeholder': 'Message 1\nMessage 2\nMessage 3...',
        'multiple_help': 'Each line will be classified as a separate message',
        'examples_info': 'Select one or more example messages to classify:',
        'classify_selected': '🔍 Classify Selected',
        'no_examples_warning': '⚠️ Select at least one example',
        
        # Results
        'results': 'Results',
        'classifying': '⏳ Classifying messages...',
        'message': 'Message',
        'predicted_category': 'Predicted Category',
        'confidence': 'Confidence',
        'low_confidence': 'Low confidence',
        'high_confidence': 'High',
        'acceptable': 'Acceptable',
        'low': 'Low',
        'no_prediction': '❌ Could not generate prediction',
        'probability_distribution': 'Probability Distribution',
        'category': 'Category',
        'threshold': 'Threshold',
        
        # Export
        'export_results': 'Export Results',
        'download_csv': 'Download CSV',
        'download_json': 'Download JSON',
        
        # Documentation
        'what_is_this': 'What is this?',
        'ethics_title': 'Ethical Considerations',
        
        # Sprint 2 NER
        'abbreviations': 'Abbreviations',
        'results': 'Results',
        
        # Placeholders
        'coming_soon': 'Coming Soon - This feature is under development',
        'see_notebook': 'See notebook',
        
        # About
        'about_text': '''
        This experimental teaching project allows building an **intelligent and explainable clinical assistant** 
        using Transformer and LLM models, with emphasis on responsible AI in healthcare.
        
        **Sprints:**
        1. Message triage (Zero-Shot)
        2. Clinical text structuring (NER → JSON)
        3. SOAP note generation with self-auditing
        4. Retrieval-augmented generation (RAG) over clinical guidelines
        ''',
    }
}


def init_session_state():
    """Initialize session state variables."""
    if 'language' not in st.session_state:
        st.session_state.language = 'es'  # Default to Spanish
    if 'current_sprint' not in st.session_state:
        st.session_state.current_sprint = 1


def render_sidebar(translations: dict, lang: str):
    """Render the sidebar with project info and navigation."""
    with st.sidebar:
        # Language selector
        st.markdown("### 🌍 " + translations['select_language'])
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🇪🇸 Español", use_container_width=True, type="primary" if lang == 'es' else "secondary"):
                st.session_state.language = 'es'
                st.rerun()
        with col2:
            if st.button("🇬🇧 English", use_container_width=True, type="primary" if lang == 'en' else "secondary"):
                st.session_state.language = 'en'
                st.rerun()
        
        st.markdown("---")
        
        # Project info
        st.markdown("### 📋 " + translations['project_info'])
        st.markdown(translations['about_text'])
        
        st.markdown("---")
        
        # Sprint navigation
        st.markdown("### 🎯 " + translations['sprint_navigation'])
        
        sprints = [
            {"num": 1, "icon": "🏥", "key": "sprint1_title", "status": "completed"},
            {"num": 2, "icon": "🔖", "key": "sprint2_title", "status": "completed"},
            {"num": 3, "icon": "📝", "key": "sprint3_title", "status": "pending"},
            {"num": 4, "icon": "💬", "key": "sprint4_title", "status": "pending"},
        ]
        
        for sprint in sprints:
            # Status indicator
            if sprint["status"] == "completed":
                status_icon = "✅"
                status_text = translations['completed']
            elif sprint["status"] == "in_progress":
                status_icon = "🔄"
                status_text = translations['in_progress']
            else:
                status_icon = "⏳"
                status_text = translations['pending']
            
            # Sprint button
            button_label = f"{sprint['icon']} Sprint {sprint['num']}: {translations[sprint['key']]}"
            is_selected = st.session_state.current_sprint == sprint['num']
            
            if st.button(
                button_label,
                key=f"sprint_{sprint['num']}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state.current_sprint = sprint['num']
                st.rerun()
            
            # Status caption
            st.caption(f"{status_icon} {status_text}")
        
        st.markdown("---")
        
        # Progress bar
        st.markdown("### 📊 " + translations['project_progress'])
        completed_sprints = sum(1 for s in sprints if s["status"] == "completed")
        progress = completed_sprints / len(sprints)
        st.progress(progress)
        st.caption(f"{completed_sprints}/{len(sprints)} sprints {translations['completed'].lower()}")
        
        st.markdown("---")
        
        # Footer
        st.caption("🎓 Universidad Peruana Cayetano Heredia")
        st.caption("💻 Built with Streamlit & Transformers")


def main():
    """Main application logic."""
    # Initialize session state
    init_session_state()
    
    # Get current language and translations
    lang = st.session_state.language
    translations = TRANSLATIONS[lang]
    
    # Render sidebar
    render_sidebar(translations, lang)
    
    # Main content
    st.title("🏥 " + translations['app_title'])
    st.caption(translations['app_subtitle'])
    
    st.markdown("---")
    
    # Render selected sprint
    current_sprint = st.session_state.current_sprint
    
    if current_sprint == 1:
        sprint_1_triaje.render(translations, lang)
    elif current_sprint == 2:
        sprint_2_ner.render(translations, lang)
    elif current_sprint == 3:
        sprint_3_soap.render(translations, lang)
    elif current_sprint == 4:
        sprint_4_rag.render(translations, lang)


if __name__ == "__main__":
    main()
