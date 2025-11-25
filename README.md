# Asistente Clínico Inteligente y Explicable - UPCH


# Asistente Clínico Inteligente y Explicable (ACIE)

Proyecto docente–experimental en el que los estudiantes construyen, en 4 sprints, un **asistente clínico inteligente y explicable** usando modelos tipo Transformer y LLM, con énfasis en:

- Triaje de mensajes de pacientes (Zero-Shot).
- Estructuración de texto clínico (NER → JSON).
- Generación responsable de notas SOAP mediante prompting avanzado.
- Recuperación aumentada por búsqueda (RAG) sobre guías clínicas.
- Integración en un MVP web y discusión ética / de seguridad.

---

## Objetivo general

Diseñar e implementar un prototipo funcional de asistente clínico que pueda:

1. Clasificar mensajes clínicos en categorías de triaje.
2. Extraer información clave de notas clínicas y convertirla en datos estructurados.
3. Redactar notas SOAP con mecanismos de auto-auditoría para reducir alucinaciones.
4. Consultar documentación clínica y responder con evidencia citada.
5. Integrar todo en una aplicación web sencilla (Gradio/Streamlit) que sirva como base para discusión sobre IA responsable en salud.

---

## Objetivos específicos

1. Comprender en profundidad la arquitectura Transformer y su rol en aplicaciones clínicas modernas.
2. Implementar un clasificador Zero-Shot clínico robusto para triaje de mensajes de pacientes.
3. Diseñar un esquema JSON clínico y un módulo de NER que convierta texto libre en datos estructurados.
4. Orquestar LLMs para generar notas SOAP, incorporando estrategias de prompting avanzado y auto-auditoría.
5. Construir un prototipo RAG que consulte guías/protocolos clínicos y devuelva respuestas citadas y auditables.
6. Desplegar un MVP web que integre todos los módulos y sirva como plataforma de discusión sobre privacidad, sesgos y gobernanza de modelos.

Proyecto organizado por sprints para construir un asistente clínico usando Transformers y LLMs.

## 📁 Estructura del Proyecto

```
asistente_clinico_upch/
├── data/                  # Datos simulados (cumpliendo privacidad)
│   ├── raw/               # Mensajes originales, PDFs de guías
│   └── processed/         # JSONs generados por el Sprint 2
├── notebooks/             # El "Laboratorio" (Google Colab)
│   ├── 1_triaje_zeroshot.ipynb      # Sprint 1
│   ├── 2_ner_estructurador.ipynb    # Sprint 2
│   ├── 3_soap_auditor.ipynb         # Sprint 3
│   └── 4_rag_chat.ipynb             # Sprint 4
├── src/                   # Código modular para el MVP (Unidad 5)
│   ├── app.py             # Entry point de Streamlit/Gradio
│   └── utils.py           # Funciones de limpieza y carga
└── requirements.txt       # Dependencias (transformers, langchain, gradio)
```

## 🎯 Sprints

### Sprint 1: Triaje Zero-Shot
Clasificación de mensajes de pacientes en categorías de urgencia sin entrenamiento adicional.

### Sprint 2: NER y Estructuración
Extracción de entidades clínicas y conversión a formato JSON estructurado.

### Sprint 3: Generador SOAP con Auto-auditoría
Generación responsable de notas SOAP con mecanismos para reducir alucinaciones.

### Sprint 4: RAG sobre Guías Clínicas
Sistema de consulta con recuperación aumentada sobre documentación médica.


## ⚠️ Privacidad y Ética

Este proyecto trabaja con datos **simulados** que cumplen con estándares de privacidad. 
Todos los datos sensibles están excluidos del control de versiones mediante `.gitignore`.

## 📝 Notas

- Los notebooks están diseñados para ejecutarse en Google Colab

- Mantener siempre la privacidad de los datos de prueba si son reales

