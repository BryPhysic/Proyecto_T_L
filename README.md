# 🏥 ACIE - Asistente Clínico Inteligente con Embeddings

Sistema de NLP Médico avanzado desarrollado para el curso **Transformers en Salud** de la UPCH. Este proyecto integra múltiples tecnologías de IA (Zero-Shot, NER, Generación de Texto, RAG) para asistir en distintas etapas del flujo de trabajo clínico.

---

## 📚 Módulos del Proyecto (Sprints)

El sistema se compone de 5 módulos principales, diseñados para evaluar distintas competencias en IA aplicada a la salud:

| Sprint | Módulo | Descripción Técnica | Notebook Evidencia |
| :--- | :--- | :--- | :--- |
| **1** | 🎯 **Gestor de Triaje** | Clasificación **Zero-Shot** de urgencias médicas (mDeBERTa-v3). Clasifica mensajes de entrada sin entrenamiento previo. | `notebooks/01_triaje_zeroshot.ipynb` |
| **2** | 🔖 **Estructurador de Datos** | Pipeline de **NER** (Named Entity Recognition) combinando HuggingFace y SciSpacy para extraer fármacos, dosis y enfermedades. | `notebooks/02_ner_basico.ipynb` |
| **3** | 📝 **Redactor Seguro** | Generador de notas **SOAP** con mecanismos de Auto-Reflexión (Self-Correction) para auditar alucinaciones. | `notebooks/05_soap_generator.ipynb` |
| **4** | 💬 **Consultor de Evidencia** | Sistema **RAG** (Retrieval-Augmented Generation) explicable. Utiliza BioMistral + PubMedBERT para responder dudas clínicas citando fuentes (PDFs). | `notebooks/04_rag_biomistral.ipynb` |
| **5** | 🚀 **Despliegue Web (MVP)** | Integración final en una Web App interactiva con **Streamlit**. Unifica todos los módulos anteriores. | `src/streamlit_app.py` |

---

## 🚀 Instalación y Ejecución

### Requisitos Previos
- Python 3.10 o superior
- [Ollama](https://ollama.ai) instalado (para ejecución local de modelos grandes en Sprint 3 y 4)
- 8GB RAM mínimo (16GB recomendado)

### 1. Clonar el Repositorio
```bash
git clone https://github.com/BryPhysic/Proyecto_T_L.git
cd Proyecto_T_L
```

### 2. Configurar Entorno Virtual
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# O en Windows: .venv\Scripts\activate
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar Ollama (Modelos Locales)
Para los módulos de Generación (Sprint 3) y RAG (Sprint 4) necesitarás los modelos base:
```bash
# Instalar Ollama (si no lo tienes)
brew install ollama  # macOS

# Descargar modelos necesarios
ollama pull llama2
ollama pull meditron:7b  # Opcional, para mejor contexto médico
```

### 5. Iniciar la Aplicación Web
```bash
streamlit run src/streamlit_app.py
```
La aplicación se abrirá automáticamente en `http://localhost:8501`.

---

## 📦 Características del Sistema

### ✅ Modo LITE (Por defecto)
- Funciona "out-of-the-box" sin configuraciones complejas.
- Permite subir tus propios documentos (PDF/TXT) para el módulo RAG.
- Usa modelos cuantizados para correr en hardware de consumo.

### 📚 Base de Conocimiento (RAG)
El sistema permite cargar Guías Clínicas y Protocolos en la carpeta `data/` o subirlos directamente desde la interfaz. El asistente usará estrictamente estos documentos para responder consultas, garantizando la trazabilidad.

---

## 📁 Estructura del Repositorio

```
Proyecto_T_L/
├── src/
│   ├── streamlit_app.py      # 🏁 Punto de entrada de la Web App
│   ├── modules/              # Lógica de cada página/sprint
│   └── utils/                # Utilidades de procesamiento (NER, RAG, PDF loader)
├── notebooks/                # 📓 Notebooks educativos (Evidencias de Evaluación)
│   ├── 01_triaje_zeroshot.ipynb
│   ├── 02_ner_basico.ipynb
│   ├── 04_rag_biomistral.ipynb
│   └── 05_soap_generator.ipynb
├── data/                     # Carpeta para documentos de conocimiento
└── requirements.txt          # Dependencias del proyecto
```

---

## 🔧 Solución de Problemas Comunes

**1. Error "Ollama connection refused"**
Asegúrate de que el servidor de Ollama esté corriendo en otra terminal:
```bash
ollama serve
```

**2. Dependencias de Spacy/SciSpacy**
Si tienes errores instalando `scispacy`, asegúrate de tener las herramientas de compilación de C++ instaladas (Xcode Command Line Tools en Mac).

---

## 👥 Créditos Académicos

Desarrollado como Proyecto Final para el curso **Transformers del Lenguaje en Salud**.
**Institución:** Universidad Peruana Cayetano Heredia (UPCH)

**Año:** 2026

---
⚠️ **Disclaimer:** Este software es una herramienta educativa y prototipo de investigación. NO debe utilizarse para toma de decisiones clínicas reales sin supervisión humana experta.
