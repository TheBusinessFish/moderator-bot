#!/usr/bin/env python3
"""
AI Content Moderator Bot for Telegram
Handles text moderation using NLP models
"""

import os
import re
import logging
from typing import Optional, Dict, Any

import torch
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message
from transformers import pipeline

# --- Configuration ---
class Config:
    """Bot configuration from environment variables"""
    BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN")
    ADMIN_ID: int = int(os.getenv("ADMIN_ID", 0))
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    TOXICITY_THRESHOLD: float = float(os.getenv("TOXICITY_THRESHOLD", 0.85))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# --- Moderation Service ---
class ModerationService:
    """Handles content analysis using NLP models"""
    
    def __init__(self):
        self._init_models()
        self.spam_patterns = [
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            r"\b\d{10,}\b"  # Phone numbers/long digit sequences
        ]

    def _init_models(self):
        """Initialize NLP models with error handling"""
        try:
            device = 0 if torch.cuda.is_available() else -1
            self.toxicity_model = pipeline(
                "text-classification",
                model="cointegrated/rubert-tiny-toxicity",
                device=device
            )
        except Exception as e:
            logging.critical(f"Failed to load model: {e}")
            raise

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for violations
        Returns: {
            'toxicity': float,
            'is_spam': bool,
            'violations': list[str]
        }
        """
        if not text.strip():
            return {"toxicity": 0.0, "is_spam": False, "violations": []}

        try:
            toxicity = self.toxicity_model(text[:1000])[0]["score"]  # Trim long texts
            is_spam = any(re.search(p, text) for p in self.spam_patterns)
            
            violations = []
            if toxicity > Config.TOXICITY_THRESHOLD:
                violations.append("toxicity")
            if is_spam:
                violations.append("spam")

            return {
                "toxicity": toxicity,
                "is_spam": is_spam,
                "violations": violations
            }
        except Exception as e:
            logging.error(f"Analysis failed: {e}")
            return {"toxicity": 0.0, "is_spam": False, "violations": []}

# --- Bot Setup ---
bot = Bot(token=Config.BOT_TOKEN)
dp = Dispatcher()
moderator = ModerationService()

# --- Handlers ---
@dp.message(Command("start"))
async def handle_start(message: Message):
    """Welcome message with bot capabilities"""
    await message.answer(
        "🛡️ Content Moderator Bot\n\n"
        "I analyze messages for:\n"
        "- Toxic language\n"
        "- Spam links\n"
        "- Scam content"
    )

@dp.message()
async def moderate_message(message: types.Message):
    """Main moderation handler"""
    try:
        if not message.text:
            return

        analysis = moderator.analyze_text(message.text)
        
        if analysis["violations"]:
            await message.delete()
            await message.answer(
                f"⚠️ Message removed. Violations: {', '.join(analysis['violations'])}\n"
                f"Toxicity score: {analysis['toxicity']:.2f}"
            )
            logging.info(f"Deleted message from {message.from_user.id}: {message.text[:50]}...")

    except Exception as e:
        logging.error(f"Moderation error: {e}")
        if Config.ADMIN_ID:
            await bot.send_message(
                Config.ADMIN_ID,
                f"🚨 Moderation failed for message: {e}"
            )

# --- Main ---
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=Config.LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        logging.info("Starting moderation bot...")
        dp.run_polling(bot)
    except Exception as e:
        logging.critical(f"Bot crashed: {e}")
        raise
