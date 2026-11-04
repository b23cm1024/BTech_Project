"""
provider_layer.py — Configurable Free-Tier LLM Provider & Quota Management Layer
================================================================================
Implements Section 9 and Section 14 of the Master Engineering Directive:
  - Connects to Google Gemini API using the official Google GenAI Python SDK
    (with google.genai and backward compatibility with google.generativeai).
  - Defaults to free-tier model: gemini-2.5-flash (or gemini-3.8-flash if configured).
  - Zero billable tools or paid grounding; zero credit spending leaks.
  - Robust error handling for 429 (ResourceExhausted / RateLimit) and 503 errors.
  - Transparent fallback to verified deterministic grounded answer synthesis
    when quota is exhausted or network is offline.
  - Supports Groq as an alternative free provider.
"""

import os
import time
from typing import Dict, Any, Optional, List


class ProviderLayer:
    """Manages LLM inference strictly within free tier boundaries."""

    DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
    DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

    @classmethod
    def generate_chat_completion(
        cls,
        system_instruction: str,
        user_prompt: str,
        provider: str = "Google Gemini",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Executes inference using free tier models with automatic error recovery.
        Returns:
            {"text": str, "provider": str, "model": str, "fallback_used": bool, "error": Optional[str]}
        """
        provider_norm = provider.strip() if provider else "Google Gemini"
        active_key = api_key or os.getenv("GEMINI_API_KEY" if provider_norm == "Google Gemini" else "GROQ_API_KEY", "")

        if not active_key:
            return {
                "text": "",
                "provider": provider_norm,
                "model": model_name or "unknown",
                "fallback_used": True,
                "error": "NO_API_KEY"
            }

        if provider_norm == "Google Gemini":
            return cls._call_gemini(
                api_key=active_key,
                system_instruction=system_instruction,
                user_prompt=user_prompt,
                model_name=model_name or cls.DEFAULT_GEMINI_MODEL,
                temperature=temperature
            )
        elif provider_norm == "Groq":
            return cls._call_groq(
                api_key=active_key,
                system_instruction=system_instruction,
                user_prompt=user_prompt,
                model_name=model_name or cls.DEFAULT_GROQ_MODEL,
                chat_history=chat_history,
                max_tokens=max_tokens,
                temperature=temperature
            )
        else:
            return {
                "text": "",
                "provider": provider_norm,
                "model": "unsupported",
                "fallback_used": True,
                "error": f"Unsupported provider {provider_norm}"
            }

    @classmethod
    def _call_gemini(
        cls,
        api_key: str,
        system_instruction: str,
        user_prompt: str,
        model_name: str,
        temperature: float
    ) -> Dict[str, Any]:
        """Inference via Google Gemini with SDK auto-selection."""
        # Try new google.genai SDK first
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            # Normalize model name for genai
            target_model = model_name
            if not target_model.startswith("gemini-"):
                target_model = cls.DEFAULT_GEMINI_MODEL

            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
            )
            response = client.models.generate_content(
                model=target_model,
                contents=user_prompt,
                config=config,
            )
            return {
                "text": response.text or "",
                "provider": "Google Gemini",
                "model": target_model,
                "fallback_used": False,
                "error": None
            }
        except Exception as e_new:
            # Fall back to google.generativeai
            try:
                import google.generativeai as genai_legacy
                genai_legacy.configure(api_key=api_key)
                target_model = model_name if "gemini" in model_name else "gemini-1.5-flash"
                model = genai_legacy.GenerativeModel(target_model, system_instruction=system_instruction)
                resp = model.generate_content(user_prompt)
                return {
                    "text": resp.text or "",
                    "provider": "Google Gemini",
                    "model": target_model,
                    "fallback_used": False,
                    "error": None
                }
            except Exception as e_leg:
                print(f"[ProviderLayer] Gemini inference exception: {e_new} / {e_leg}")
                return {
                    "text": "",
                    "provider": "Google Gemini",
                    "model": model_name,
                    "fallback_used": True,
                    "error": str(e_new or e_leg)
                }

    @classmethod
    def _call_groq(
        cls,
        api_key: str,
        system_instruction: str,
        user_prompt: str,
        model_name: str,
        chat_history: Optional[List[Dict[str, str]]],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Inference via Groq with fast timeout and retry prevention."""
        try:
            from groq import Groq
            client = Groq(api_key=api_key, timeout=12.0, max_retries=0)
            messages = [{"role": "system", "content": system_instruction}]

            if chat_history:
                for msg in chat_history[-3:]:
                    content = msg.get("content", "")
                    if msg.get("role") == "assistant" and len(content) > 250:
                        content = content[:250] + "..."
                    messages.append({"role": msg.get("role", "user"), "content": content})

            messages.append({"role": "user", "content": user_prompt})

            sel_model = model_name or cls.DEFAULT_GROQ_MODEL
            try:
                resp = client.chat.completions.create(
                    model=sel_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return {
                    "text": resp.choices[0].message.content or "",
                    "provider": "Groq",
                    "model": sel_model,
                    "fallback_used": False,
                    "error": None
                }
            except Exception as groq_err:
                fallback_m = "openai/gpt-oss-20b" if "120b" in sel_model else "qwen/qwen3.8-27b"
                resp = client.chat.completions.create(
                    model=fallback_m,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return {
                    "text": resp.choices[0].message.content or "",
                    "provider": "Groq",
                    "model": fallback_m,
                    "fallback_used": False,
                    "error": None
                }
        except Exception as e:
            print(f"[ProviderLayer] Groq inference exception: {e}")
            return {
                "text": "",
                "provider": "Groq",
                "model": model_name,
                "fallback_used": True,
                "error": str(e)
            }
