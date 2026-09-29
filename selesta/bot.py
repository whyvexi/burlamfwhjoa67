import asyncio
import logging

import aiosqlite
import httpx
from aiogram import BaseMiddleware, Bot, Dispatcher, Router
from aiogram.enums import ChatType
from aiogram.types import BotCommand, BufferedInputFile, Message
from aiogram.utils.chat_action import ChatActionSender

from .config import Config, ROOT
from .llm import LLM, LLMError, SYSTEM
from .storage import Store

log = logging.getLogger(__name__)
HELP = '''Selesta — твоя техническая помощница.

Отправь текст: разберём код, архитектуру, монтаж или текущую задачу.

/reset — очистить историю диалога
/remember текст — сохранить постоянную заметку
/memory — показать заметки
/forget ID — удалить заметку
/profile — твой базовый профиль
/status — модель и размер истории
/search запрос — поиск в интернете
/read URL — прочитать страницу
/id — твой Telegram ID
/help — команды

Профиль и заметки сохраняются после /reset. Фото, голос и вложения пока не обрабатываю.'''


def command_parts(text: str):
    head, _, tail = text.partition(' ')
    return head.split('@', 1)[0].lower(), tail.strip()


def split_text(text: str, limit: int = 3500):
    # 3500 UTF-16 units leaves room below Telegram's 4096-character limit.
    chunk, size = [], 0
    for char in text:
        units = 2 if ord(char) > 0xFFFF else 1
        if size + units > limit:
            yield ''.join(chunk)
            chunk, size = [], 0
        chunk.append(char)
        size += units
    if chunk:
        yield ''.join(chunk)


async def send_text(message: Message, text: str):
    if len(text) > 12000:
        await message.answer_document(
            BufferedInputFile(text.encode('utf-8'), filename='selesta-answer.md'),
            caption='Полный ответ — в файле, чтобы сохранить код и форматирование.',
        )
    else:
        for chunk in split_text(text):
            await message.answer(chunk, parse_mode=None)


class OwnerOnly(BaseMiddleware):
    def __init__(self, owner_id: int | None):
        self.owner_id = owner_id

    async def __call__(self, handler, event: Message, data):
        if event.chat.type != ChatType.PRIVATE or event.from_user is None:
            return
        command, _ = command_parts(event.text or '')
        if command == '/id':
            await event.answer(f'Твой Telegram ID: {event.from_user.id}', parse_mode=None)
            return
        if self.owner_id is None or event.from_user.id != self.owner_id:
            await event.answer('Доступ закрыт. /id — узнать свой Telegram ID.', parse_mode=None)
            return
        return await handler(event, data)


def build_router(config: Config, store: Store, llm: LLM):
    router = Router()
    router.message.outer_middleware(OwnerOnly(config.owner_id))
    # One owner, one conversation. /reset and memory edits also wait for generation.
    lock = asyncio.Lock()

    @router.message()
    async def handle(message: Message):
        if not message.text:
            await message.answer('Пока принимаю только текст. Пришли вопрос или код сообщением.')
            return
        async with lock:
            text = message.text
            command, argument = command_parts(text)
            if command in {'/start', '/help'}:
                await message.answer(HELP)
            elif command == '/reset':
                await store.reset()
                await message.answer('История очищена. Профиль и постоянные заметки сохранены.')
            elif command == '/profile':
                await send_text(message, (ROOT / 'profile.md').read_text(encoding='utf-8'))
            elif command == '/remember':
                if not argument:
                    await message.answer('Пример: /remember Для новых ботов используй Python 3.12')
                    return
                try:
                    memory_id = await store.remember(argument)
                except ValueError as exc:
                    await message.answer(str(exc))
                else:
                    await message.answer(f'Сохранено. ID заметки: {memory_id}')
            elif command == '/memory':
                notes = await store.memories()
                await send_text(message, '\n\n'.join(f'{i}. {note}' for i, note in notes) or 'Постоянных заметок пока нет.')
            elif command == '/forget':
                if not argument.isdecimal() or len(argument) > 18:
                    await message.answer('Формат: /forget ID. Список ID: /memory')
                else:
                    removed = await store.forget(int(argument))
                    await message.answer('Заметка удалена. Для удаления из контекста переписки также используй /reset.' if removed else 'Заметка не найдена.')
            elif command == '/status':
                await message.answer(
                    f'Модель: {config.model}\nAPI: {config.base_url}\n'
                    f'История: до {config.history_turns} пар сообщений\n'
                    f'Бюджет истории: {config.context_chars} символов\n'
                    'Режим: текст, профиль, память, веб-поиск и чтение ссылок'
                )
            elif text.startswith('/') and command not in {'/search', '/read'}:
                await message.answer('Неизвестная команда. Список: /help')
            else:
                initial_tool = None
                if command in {'/search', '/read'}:
                    if not argument:
                        await message.answer('Формат: /search запрос или /read https://example.com')
                        return
                    initial_tool = ('web_search', {'query': argument}) if command == '/search' else ('read_url', {'url': argument})
                async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
                    profile = (ROOT / 'profile.md').read_text(encoding='utf-8')
                    notes = await store.memories()
                    system = SYSTEM + '\nПРОФИЛЬ:\n' + profile
                    if notes:
                        system += '\nПОСТОЯННЫЕ ЗАМЕТКИ:\n' + '\n'.join(note for _, note in notes)
                    history = await store.history(max(0, config.context_chars - len(text)))
                    try:
                        answer = await llm.answer([
                            {'role': 'system', 'content': system},
                            *history,
                            {'role': 'user', 'content': text},
                        ], initial_tool=initial_tool)
                    except LLMError as exc:
                        await message.answer(str(exc))
                        return
                    # Persist before delivery so a Telegram delivery failure doesn't lose a paid answer.
                    await store.save_turn(text, answer)
                    await send_text(message, answer)

    return router


async def main():
    config = Config.load()
    (ROOT / 'profile.md').read_text(encoding='utf-8')
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    async with (
        aiosqlite.connect(data / 'selesta.db') as db,
        httpx.AsyncClient(timeout=httpx.Timeout(config.timeout, connect=15)) as client,
        Bot(token=config.telegram_token) as bot,
    ):
        store = Store(db, config.history_turns)
        await store.initialize()
        dispatcher = Dispatcher()
        dispatcher.include_router(build_router(config, store, LLM(config, client)))

        @dispatcher.errors()
        async def on_error(event):
            # Do not log exception bodies: HTTP errors can contain secrets or message text.
            log.error('Ошибка обработки обновления: %s', type(event.exception).__name__)
            message = event.update.message
            if message and message.chat.type == ChatType.PRIVATE and message.from_user and message.from_user.id == config.owner_id:
                try:
                    await message.answer('Ошибка обработки сообщения. Проверь журнал бота и попробуй ещё раз.')
                except Exception:
                    log.error('Не удалось отправить уведомление об ошибке')
            return True

        await bot.set_my_commands([
            BotCommand(command=command, description=description)
            for command, description in [
                ('help', 'Команды Selesta'), ('reset', 'Очистить историю'),
                ('remember', 'Сохранить заметку'), ('memory', 'Постоянные заметки'),
                ('forget', 'Удалить заметку по ID'), ('profile', 'Базовый профиль'),
                ('status', 'Настройки модели'), ('id', 'Мой Telegram ID'),
                ('search', 'Поиск в интернете'), ('read', 'Прочитать страницу по URL'),
            ]
        ])
        # Long polling requires removing a previous webhook; queued messages are retained.
        await bot.delete_webhook(drop_pending_updates=False)
        log.info('Selesta запущена. Доступ: %s', 'владелец' if config.owner_id else 'только /id')
        await dispatcher.start_polling(bot, allowed_updates=['message'], close_bot_session=False)
