import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    telegram_token: str = field(repr=False)
    api_key: str = field(repr=False)
    owner_id: int | None
    base_url: str
    model: str
    timeout: int
    max_tokens: int
    history_turns: int
    context_chars: int

    @classmethod
    def load(cls):
        load_dotenv(ROOT / '.env')

        def required(name):
            value = os.getenv(name, '').strip()
            if not value:
                raise ValueError(f'Заполни {name} в .env')
            return value

        # Bothost supplies BOT_TOKEN; do not use its USER_ID as a Telegram owner ID.
        telegram_token = next((os.getenv(name, '').strip() for name in
                               ('TELEGRAM_BOT_TOKEN', 'BOT_TOKEN', 'API_TOKEN')
                               if os.getenv(name, '').strip()), '')
        if not telegram_token:
            raise ValueError('Заполни TELEGRAM_BOT_TOKEN или BOT_TOKEN в .env/переменных окружения хостинга')

        def integer(name, default, low, high):
            try:
                value = int(os.getenv(name, str(default)))
            except ValueError:
                raise ValueError(f'{name}: требуется целое число') from None
            if not low <= value <= high:
                raise ValueError(f'{name}: допустимо от {low} до {high}')
            return value

        owner = os.getenv('OWNER_TELEGRAM_ID', '').strip()
        if owner and (not owner.isdecimal() or int(owner) <= 0):
            raise ValueError('OWNER_TELEGRAM_ID: нужен положительный числовой Telegram ID')
        base = os.getenv('LLM_BASE_URL', 'https://plusvibeapi.ru/v1').strip().rstrip('/')
        parsed = urlparse(base)
        if parsed.scheme != 'https' or not parsed.netloc or parsed.query or parsed.fragment or parsed.username:
            raise ValueError('LLM_BASE_URL: нужен HTTPS URL без логина, query и fragment')
        return cls(
            telegram_token=telegram_token,
            api_key=required('LLM_API_KEY'),
            owner_id=int(owner) if owner else None,
            base_url=base,
            model=os.getenv('LLM_MODEL', 'deepseek/deepseek-v4-flash-0731').strip(),
            timeout=integer('LLM_TIMEOUT', 120, 10, 600),
            max_tokens=integer('LLM_MAX_TOKENS', 4096, 256, 32768),
            history_turns=integer('HISTORY_TURNS', 20, 1, 100),
            context_chars=integer('CONTEXT_CHARS', 40000, 4000, 200000),
        )
