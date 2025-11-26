"""
Triaje Clinical Classifier Module
==================================
Encapsulates the Zero-Shot classification model for clinical triage.
Uses MoritzLaurer/mDeBERTa-v3-base-mnli-xnli for multilingual zero-shot classification.
"""

import streamlit as st
from transformers import pipeline
from typing import List, Dict, Any, Optional


class TriajeClassifier:
    """
    Clinical triage classifier using zero-shot classification.
    
    This class provides methods to classify clinical messages into
    predefined or custom triage categories using a multilingual model.
    """
    
    MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    
    # Predefined labels
    LABELS_ES = [
        "Urgencia Vital (Roja)",
        "Consulta Administrativa",
        "Renovación de Receta",
        "Consulta Médica No Urgente"
    ]
    
    LABELS_EN = [
        "Life-Threatening Emergency (Red)",
        "Administrative Inquiry",
        "Prescription Renewal",
        "Non-Urgent Medical Consultation"
    ]
    
    # Example messages
    EXAMPLES_ES = [
        "Tengo un dolor muy fuerte en el pecho y se me duerme el brazo izquierdo.",
        "Hola, necesito saber si el doctor Martínez atiende los jueves.",
        "Se me acabó la pastilla de la presión, necesito la receta para comprarla.",
        "Mi hijo tiene fiebre de 38 desde ayer y llora mucho, pero come bien.",
        "Necesito que me renueven el enalapril que se me terminó ayer",
        "Siento una presión muy fuerte en el pecho y me falta el aire"
    ]
    
    EXAMPLES_EN = [
        "I have very severe chest pain and my left arm feels numb.",
        "Hello, I need to know if Dr. Martínez sees patients on Thursdays.",
        "I ran out of my blood pressure pills; I need the prescription to buy them.",
        "My child has had a 38°C fever since yesterday and cries a lot, but is eating well.",
        "I need my enalapril renewed; I ran out yesterday.",
        "I feel very strong pressure in my chest and I'm short of breath."
    ]
    
    def __init__(self):
        """Initialize the classifier by loading the model."""
        self.classifier = self._load_model()
    
    @staticmethod
    @st.cache_resource
    def _load_model():
        """
        Load the zero-shot classification model with caching.
        
        Returns:
            pipeline: Hugging Face pipeline for zero-shot classification
        """
        return pipeline(
            "zero-shot-classification",
            model=TriajeClassifier.MODEL_NAME
        )
    
    def classify(
        self,
        message: str,
        labels: List[str],
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Classify a single message into one of the provided labels.
        
        Args:
            message: The clinical message to classify
            labels: List of possible triage categories
            threshold: Optional confidence threshold (0.0-1.0)
        
        Returns:
            Dictionary containing:
                - predicted_label: The top predicted category
                - confidence: Confidence score for the prediction
                - all_scores: Dictionary with all labels and their scores
                - below_threshold: Boolean indicating if confidence is below threshold
        """
        if not message.strip():
            return {
                "predicted_label": None,
                "confidence": 0.0,
                "all_scores": {},
                "below_threshold": True
            }
        
        # Run classification
        result = self.classifier(message, labels)
        
        # Extract results
        predicted_label = result['labels'][0]
        confidence = result['scores'][0]
        
        # Create dictionary of all scores
        all_scores = dict(zip(result['labels'], result['scores']))
        
        # Check threshold
        below_threshold = False
        if threshold is not None:
            below_threshold = confidence < threshold
        
        return {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "all_scores": all_scores,
            "below_threshold": below_threshold,
            "message": message
        }
    
    def classify_batch(
        self,
        messages: List[str],
        labels: List[str],
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Classify multiple messages.
        
        Args:
            messages: List of clinical messages to classify
            labels: List of possible triage categories
            threshold: Optional confidence threshold
        
        Returns:
            List of classification results
        """
        results = []
        for message in messages:
            result = self.classify(message, labels, threshold)
            results.append(result)
        return results
    
    @classmethod
    def get_default_labels(cls, language: str = 'es') -> List[str]:
        """
        Get predefined triage labels in the specified language.
        
        Args:
            language: 'es' for Spanish or 'en' for English
        
        Returns:
            List of triage category labels
        """
        if language.lower() == 'en':
            return cls.LABELS_EN.copy()
        return cls.LABELS_ES.copy()
    
    @classmethod
    def get_example_messages(cls, language: str = 'es') -> List[str]:
        """
        Get example messages in the specified language.
        
        Args:
            language: 'es' for Spanish or 'en' for English
        
        Returns:
            List of example clinical messages
        """
        if language.lower() == 'en':
            return cls.EXAMPLES_EN.copy()
        return cls.EXAMPLES_ES.copy()
    
    @staticmethod
    def get_urgency_color(label: str) -> str:
        """
        Get color code based on urgency level.
        
        Args:
            label: The triage category label
        
        Returns:
            Hex color code
        """
        label_lower = label.lower()
        
        if 'vital' in label_lower or 'emergency' in label_lower or 'roja' in label_lower or 'red' in label_lower:
            return '#dc2626'  # Red for critical
        elif 'renovación' in label_lower or 'renewal' in label_lower or 'receta' in label_lower or 'prescription' in label_lower:
            return '#f59e0b'  # Amber for prescription
        elif 'administrativa' in label_lower or 'administrative' in label_lower:
            return '#3b82f6'  # Blue for administrative
        else:
            return '#10b981'  # Green for non-urgent medical
    
    @staticmethod
    def format_confidence(confidence: float) -> str:
        """
        Format confidence score as percentage.
        
        Args:
            confidence: Confidence score (0.0-1.0)
        
        Returns:
            Formatted string with percentage
        """
        return f"{confidence * 100:.1f}%"
