"""
Gemini-based fake news detection service.
Uses Google's Gemini AI model for more sophisticated fake news detection.
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


@dataclass
class GeminiConfig:
    """Configuration for Gemini model."""
    api_key: str
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.1
    max_output_tokens: int = 1024


class GeminiFakeNewsDetector:
    """Fake news detector using Google's Gemini model."""
    
    def __init__(self, config: GeminiConfig):
        self.config = config
        genai.configure(api_key=config.api_key)
        self.model = genai.GenerativeModel(
            model_name=config.model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
    
    def _create_prompt(self, text: str) -> str:
        """Create a comprehensive prompt for fake news detection."""
        return f"""
You are an expert fact-checker and journalist. Analyze the following text for signs of fake news or misinformation.

Consider these factors:
1. **Source credibility**: Does it cite reliable sources?
2. **Language patterns**: Is the language sensational, inflammatory, or overly dramatic?
3. **Factual claims**: Are there verifiable facts or just opinions?
4. **Writing quality**: Is it well-written with proper grammar and structure?
5. **Emotional manipulation**: Does it use fear, anger, or other emotions to manipulate?
6. **Consistency**: Are the claims internally consistent?
7. **Timeliness**: Are the claims current and relevant?

Text to analyze:
"{text}"

Please provide your analysis in the following JSON format:
{{
    "is_fake": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of your decision",
    "red_flags": ["List of specific concerns if any"],
    "source_quality": "poor/fair/good/excellent",
    "factual_accuracy": "low/medium/high"
}}

Be objective and base your analysis on journalistic standards and fact-checking principles.
"""

    def detect_fake_news(self, text: str) -> Dict[str, Any]:
        """
        Detect if the given text is fake news using Gemini.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with detection results
        """
        try:
            prompt = self._create_prompt(text)
            response = self.model.generate_content(prompt)
            
            # Parse the JSON response
            result_text = response.text.strip()
            
            # Try to extract JSON from the response
            if "```json" in result_text:
                json_start = result_text.find("```json") + 7
                json_end = result_text.find("```", json_start)
                json_text = result_text[json_start:json_end].strip()
            elif "{" in result_text and "}" in result_text:
                json_start = result_text.find("{")
                json_end = result_text.rfind("}") + 1
                json_text = result_text[json_start:json_end]
            else:
                # Fallback parsing
                return self._fallback_analysis(text, result_text)
            
            result = json.loads(json_text)
            
            # Ensure required fields exist
            return {
                "is_fake": result.get("is_fake", False),
                "confidence": float(result.get("confidence", 0.5)),
                "reasoning": result.get("reasoning", "Analysis completed"),
                "red_flags": result.get("red_flags", []),
                "source_quality": result.get("source_quality", "unknown"),
                "factual_accuracy": result.get("factual_accuracy", "unknown"),
                "raw_response": result_text
            }
            
        except json.JSONDecodeError as e:
            return self._fallback_analysis(text, response.text if 'response' in locals() else "JSON parsing failed")
        except Exception as e:
            return {
                "is_fake": False,
                "confidence": 0.0,
                "reasoning": f"Error in analysis: {str(e)}",
                "red_flags": [],
                "source_quality": "unknown",
                "factual_accuracy": "unknown",
                "raw_response": str(e)
            }
    
    def _fallback_analysis(self, text: str, response_text: str) -> Dict[str, Any]:
        """Fallback analysis when JSON parsing fails."""
        # Simple keyword-based fallback
        fake_indicators = [
            "breaking", "shocking", "you won't believe", "doctors hate", 
            "one weird trick", "click here", "share this", "urgent",
            "exclusive", "insider", "leaked", "exposed"
        ]
        
        text_lower = text.lower()
        fake_count = sum(1 for indicator in fake_indicators if indicator in text_lower)
        
        is_fake = fake_count > 2
        confidence = min(0.8, fake_count * 0.2)
        
        return {
            "is_fake": is_fake,
            "confidence": confidence,
            "reasoning": f"Fallback analysis: found {fake_count} potential fake news indicators",
            "red_flags": [ind for ind in fake_indicators if ind in text_lower],
            "source_quality": "unknown",
            "factual_accuracy": "unknown",
            "raw_response": response_text
        }
    
    def batch_detect(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Detect fake news for multiple texts."""
        return [self.detect_fake_news(text) for text in texts]


def create_gemini_detector(api_key: Optional[str] = None) -> GeminiFakeNewsDetector:
    """
    Create a Gemini detector instance.
    
    Args:
        api_key: Gemini API key. If None, will try to get from environment.
        
    Returns:
        GeminiFakeNewsDetector instance
    """
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not provided. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
    
    config = GeminiConfig(api_key=api_key)
    return GeminiFakeNewsDetector(config)


# Convenience functions for compatibility with existing code
def predict_fake_news_gemini(texts: List[str], api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Predict fake news using Gemini for a list of texts."""
    detector = create_gemini_detector(api_key)
    return detector.batch_detect(texts)


def predict_single_gemini(text: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Predict fake news using Gemini for a single text."""
    detector = create_gemini_detector(api_key)
    return detector.detect_fake_news(text)
