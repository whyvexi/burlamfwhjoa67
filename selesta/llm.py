import httpx

from .config import Config

SYSTEM = '''Ты Selesta (Селеста), персональная ИИ-помощница Дани (eskalane).
Говори о себе в женском роде. По умолчанию отвечай на русском.
Учитывай профиль ниже. Текущие пожелания пользователя приоритетнее старых предпочтений.
Ты работаешь в Telegram: доступны текстовые ответы и явная память через команды бота.
У тебя нет инструментов поиска, терминала, доступа к файлам пользователя или выполнения
кода. Не утверждай, что запускала код, проверяла сайты или выполнила внешнее действие.
Не выдумывай факты и результаты проверок. Если данных не хватает, обозначь допущение
или задай один конкретный вопрос. Код пиши полностью для поставленной задачи.
Не утверждай, что сохранила постоянную память: для этого пользователь использует /remember.
'''


class LLMError(Exception):
    pass


class LLM:
    def __init__(self, config: Config, client: httpx.AsyncClient):
        self.config = config
        self.client = client

    async def answer(self, messages: list[dict]) -> str:
        try:
            response = await self.client.post(
                self.config.base_url + '/chat/completions',
                headers={'Authorization': f'Bearer {self.config.api_key}'},
                json={
                    'model': self.config.model,
                    'messages': messages,
                    'max_tokens': self.config.max_tokens,
                    'stream': False,
                },
            )
        except httpx.TimeoutException:
            raise LLMError('API не ответил вовремя. Попробуй ещё раз.') from None
        except httpx.RequestError:
            raise LLMError('Не удалось подключиться к API. Проверь сеть и LLM_BASE_URL.') from None
        if response.is_error:
            hints = {
                400: 'API отклонил параметры запроса. Проверь модель и LLM_MAX_TOKENS.',
                401: 'API отклонил ключ. Проверь LLM_API_KEY.',
                402: 'Недостаточно средств на балансе API.',
                403: 'API запретил доступ. Проверь права ключа и доступ к модели.',
                404: 'Endpoint или модель не найдены. Проверь LLM_BASE_URL и LLM_MODEL.',
                429: 'Лимит API исчерпан. Попробуй позже.',
            }
            raise LLMError(hints.get(response.status_code, f'Ошибка API (HTTP {response.status_code}). Попробуй позже.'))
        try:
            choice = response.json()['choices'][0]
            content = choice['message']['content']
            if not isinstance(content, str) or not content.strip():
                raise ValueError
        except (ValueError, KeyError, IndexError, TypeError):
            raise LLMError('API вернул пустой ответ или неожиданный формат.') from None
        if choice.get('finish_reason') == 'length':
            content += '\n\n[Ответ обрезан лимитом токенов. Попроси продолжить.]'
        return content
