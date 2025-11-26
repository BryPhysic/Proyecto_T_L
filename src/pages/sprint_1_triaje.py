"""
Sprint 1: Triaje Zero-Shot
Main page for clinical triage classification using zero-shot learning.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import List
import json
import io

from utils.triaje_model import TriajeClassifier


def render(translations: dict, lang: str):
    """
    Render the Sprint 1 triage page.
    
    Args:
        translations: Dictionary with UI translations
        lang: Current language code ('es' or 'en')
    """
    st.title("🏥 Sprint 1: " + translations['sprint1_title'])
    
    # Initialize classifier
    classifier = TriajeClassifier()
    
    # Documentation section
    render_documentation(translations, lang)
    
    st.markdown("---")
    
    # Configuration section
    st.subheader("⚙️ " + translations['configuration'])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Label selection
        use_custom_labels = st.checkbox(
            translations['use_custom_labels'],
            value=False,
            help=translations['custom_labels_help']
        )
        
        if use_custom_labels:
            custom_labels_input = st.text_input(
                translations['enter_labels'],
                placeholder=translations['labels_placeholder'],
                help=translations['labels_help']
            )
            labels = [l.strip() for l in custom_labels_input.split(',') if l.strip()]
            
            if not labels:
                st.warning(translations['no_labels_warning'])
        else:
            labels = classifier.get_default_labels(lang)
            st.info(f"**{translations['default_labels']}:** " + ", ".join(f"`{l}`" for l in labels))
    
    with col2:
        # Confidence threshold
        threshold = st.slider(
            translations['confidence_threshold'],
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help=translations['threshold_help']
        )
        st.caption(f"🎯 {translations['current_threshold']}: **{threshold:.0%}**")
    
    st.markdown("---")
    
    # Input section
    st.subheader("✍️ " + translations['input_messages'])
    
    # Tabs for different input methods
    tab1, tab2, tab3 = st.tabs([
        "📝 " + translations['single_message'],
        "📄 " + translations['multiple_messages'],
        "🔍 " + translations['examples']
    ])
    
    messages_to_classify = []
    
    with tab1:
        single_message = st.text_area(
            translations['enter_message'],
            height=100,
            placeholder=translations['message_placeholder']
        )
        
        if st.button(translations['classify_button'], type="primary", key="classify_single"):
            if single_message.strip():
                messages_to_classify = [single_message]
            else:
                st.warning(translations['empty_message_warning'])
    
    with tab2:
        multiple_messages = st.text_area(
            translations['enter_multiple'],
            height=200,
            placeholder=translations['multiple_placeholder'],
            help=translations['multiple_help']
        )
        
        if st.button(translations['classify_button'], type="primary", key="classify_multiple"):
            if multiple_messages.strip():
                messages_to_classify = [m.strip() for m in multiple_messages.split('\n') if m.strip()]
            else:
                st.warning(translations['empty_message_warning'])
    
    with tab3:
        example_messages = classifier.get_example_messages(lang)
        
        st.info(translations['examples_info'])
        
        selected_examples = []
        for i, example in enumerate(example_messages):
            if st.checkbox(f"{i+1}. {example}", key=f"example_{i}"):
                selected_examples.append(example)
        
        if st.button(translations['classify_selected'], type="primary", key="classify_examples"):
            if selected_examples:
                messages_to_classify = selected_examples
            else:
                st.warning(translations['no_examples_warning'])
    
    # Classification section
    if messages_to_classify and labels:
        st.markdown("---")
        st.subheader("📊 " + translations['results'])
        
        with st.spinner(translations['classifying']):
            results = classifier.classify_batch(messages_to_classify, labels, threshold)
        
        # Display results
        for i, result in enumerate(results):
            render_classification_result(
                result, i + 1, threshold, translations, lang
            )
        
        # Export section
        st.markdown("---")
        render_export_section(results, translations)
    
    # Ethical considerations
    st.markdown("---")
    render_ethics_section(translations, lang)


def render_documentation(translations: dict, lang: str):
    """Render the documentation section."""
    with st.expander("📚 " + translations['what_is_this']):
        if lang == 'es':
            st.markdown("""
            ### ¿Qué es el Triaje Zero-Shot?
            
            El **triaje zero-shot** es una técnica de clasificación de texto que permite categorizar mensajes 
            clínicos sin necesidad de entrenar un modelo específico para esta tarea.
            
            #### ¿Cómo funciona?
            
            1. **Modelo base**: Utilizamos `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, un modelo multilingüe 
               entrenado en inferencia de lenguaje natural (NLI)
            2. **Sin entrenamiento adicional**: El modelo puede clasificar en categorías que nunca ha visto antes
            3. **Flexible**: Puedes usar categorías predefinidas o crear las tuyas propias
            
            #### Ventajas
            
            - ✅ No requiere datos de entrenamiento
            - ✅ Funciona en múltiples idiomas (español, inglés, etc.)
            - ✅ Adaptable a diferentes categorías de triaje
            - ✅ Proporciona scores de confianza para cada categoría
            
            #### Limitaciones
            
            - ⚠️ Puede tener menor precisión que modelos entrenados específicamente
            - ⚠️ Requiere etiquetas bien descritas y diferenciadas
            - ⚠️ No reemplaza el juicio clínico profesional
            """)
        else:
            st.markdown("""
            ### What is Zero-Shot Triage?
            
            **Zero-shot triage** is a text classification technique that allows categorizing clinical 
            messages without the need to train a specific model for this task.
            
            #### How does it work?
            
            1. **Base model**: We use `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, a multilingual model 
               trained on natural language inference (NLI)
            2. **No additional training**: The model can classify into categories it has never seen before
            3. **Flexible**: You can use predefined categories or create your own
            
            #### Advantages
            
            - ✅ Requires no training data
            - ✅ Works in multiple languages (Spanish, English, etc.)
            - ✅ Adaptable to different triage categories
            - ✅ Provides confidence scores for each category
            
            #### Limitations
            
            - ⚠️ May have lower accuracy than specifically trained models
            - ⚠️ Requires well-described and differentiated labels
            - ⚠️ Does not replace professional clinical judgment
            """)


def render_classification_result(
    result: dict,
    index: int,
    threshold: float,
    translations: dict,
    lang: str
):
    """Render a single classification result."""
    with st.container():
        st.markdown(f"#### {translations['message']} {index}")
        
        # Display message
        st.text(result['message'])
        
        if result['predicted_label']:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Predicted label with color
                color = TriajeClassifier.get_urgency_color(result['predicted_label'])
                confidence_pct = TriajeClassifier.format_confidence(result['confidence'])
                
                # Warning if below threshold
                if result['below_threshold']:
                    st.warning(
                        f"⚠️ {translations['low_confidence']}: **{confidence_pct}** < {threshold:.0%}"
                    )
                
                st.markdown(
                    f"**{translations['predicted_category']}:** "
                    f"<span style='color: {color}; font-weight: bold;'>{result['predicted_label']}</span> "
                    f"({confidence_pct})",
                    unsafe_allow_html=True
                )
            
            with col2:
                # Confidence indicator
                if result['confidence'] >= 0.7:
                    st.success(f"🎯 {translations['high_confidence']}")
                elif result['confidence'] >= threshold:
                    st.info(f"✓ {translations['acceptable']}")
                else:
                    st.error(f"⚠️ {translations['low']}")
            
            # Probability chart
            if result['all_scores']:
                fig = create_probability_chart(
                    result['all_scores'],
                    threshold,
                    translations
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(translations['no_prediction'])
        
        st.markdown("---")


def create_probability_chart(
    scores: dict,
    threshold: float,
    translations: dict
) -> go.Figure:
    """Create a horizontal bar chart for probabilities."""
    # Sort by score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_scores]
    values = [item[1] for item in sorted_scores]
    
    # Color bars
    colors = [TriajeClassifier.get_urgency_color(label) for label in labels]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=labels,
        x=values,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='#333', width=1)
        ),
        text=[f"{v:.1%}" for v in values],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>' + translations['confidence'] + ': %{x:.1%}<extra></extra>'
    ))
    
    # Add threshold line
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"{translations['threshold']}: {threshold:.0%}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=translations['probability_distribution'],
        xaxis_title=translations['confidence'],
        yaxis_title=translations['category'],
        xaxis=dict(tickformat='.0%', range=[0, 1]),
        height=max(300, len(labels) * 60),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def render_export_section(results: List[dict], translations: dict):
    """Render the export section for results."""
    st.subheader("💾 " + translations['export_results'])
    
    col1, col2 = st.columns(2)
    
    # Prepare data for export
    export_data = []
    for i, result in enumerate(results):
        export_data.append({
            'message_id': i + 1,
            'message': result['message'],
            'predicted_label': result['predicted_label'],
            'confidence': result['confidence'],
            'below_threshold': result['below_threshold'],
            **{f'score_{label}': score for label, score in result['all_scores'].items()}
        })
    
    with col1:
        # CSV export
        df = pd.DataFrame(export_data)
        csv = df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 " + translations['download_csv'],
            data=csv,
            file_name="triaje_results.csv",
            mime="text/csv"
        )
    
    with col2:
        # JSON export
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        
        st.download_button(
            label="📥 " + translations['download_json'],
            data=json_str,
            file_name="triaje_results.json",
            mime="application/json"
        )


def render_ethics_section(translations: dict, lang: str):
    """Render ethical considerations section."""
    with st.expander("⚖️ " + translations['ethics_title']):
        if lang == 'es':
            st.markdown("""
            ### Consideraciones Éticas y Limitaciones
            
            #### ⚠️ Este sistema NO reemplaza el juicio clínico profesional
            
            - El modelo es una **herramienta de apoyo**, no un sistema de diagnóstico
            - Todas las clasificaciones deben ser **validadas por personal médico calificado**
            - En casos de urgencia vital, **siempre priorizar el protocolo clínico establecido**
            
            #### 🔒 Privacidad y Seguridad
            
            - No ingresar información personal identificable (nombres, números de documento, etc.)
            - Los datos procesados en esta demo son temporales y no se almacenan
            - En un sistema de producción, se debe cumplir con HIPAA, GDPR y regulaciones locales
            
            #### 🎯 Limitaciones Técnicas
            
            - **Precisión variable**: El modelo puede cometer errores, especialmente con:
              - Mensajes ambiguos o poco claros
              - Terminología médica muy específica
              - Casos límite entre categorías
            - **Contexto limitado**: El modelo solo analiza el texto proporcionado, sin historial clínico
            - **Sesgos potenciales**: El modelo puede tener sesgos inherentes de sus datos de entrenamiento
            
            #### 📊 Recomendaciones de Uso
            
            1. Usar **múltiples categorías bien diferenciadas**
            2. Ajustar el **threshold de confianza** según el contexto de uso
            3. **Revisar manualmente** casos con baja confianza
            4. **Documentar** decisiones que difieran de la clasificación automática
            5. **Evaluar continuamente** el rendimiento del sistema con datos reales
            """)
        else:
            st.markdown("""
            ### Ethical Considerations and Limitations
            
            #### ⚠️ This system does NOT replace professional clinical judgment
            
            - The model is a **support tool**, not a diagnostic system
            - All classifications must be **validated by qualified medical personnel**
            - In life-threatening emergencies, **always prioritize established clinical protocols**
            
            #### 🔒 Privacy and Security
            
            - Do not enter personally identifiable information (names, ID numbers, etc.)
            - Data processed in this demo is temporary and not stored
            - In a production system, must comply with HIPAA, GDPR, and local regulations
            
            #### 🎯 Technical Limitations
            
            - **Variable accuracy**: The model may make errors, especially with:
              - Ambiguous or unclear messages
              - Highly specific medical terminology
              - Borderline cases between categories
            - **Limited context**: The model only analyzes the provided text, without clinical history
            - **Potential biases**: The model may have inherent biases from its training data
            
            #### 📊 Usage Recommendations
            
            1. Use **multiple well-differentiated categories**
            2. Adjust **confidence threshold** according to use context
            3. **Manually review** low-confidence cases
            4. **Document** decisions that differ from automatic classification
            5. **Continuously evaluate** system performance with real data
            """)
