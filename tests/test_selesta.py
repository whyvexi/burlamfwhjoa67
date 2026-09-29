import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiosqlite
import httpx

from selesta.bot import OwnerOnly, split_text
from selesta.config import Config
from selesta.llm import LLM, LLMError
from selesta.storage import Store


class StorageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / 'test.db'
        self.db = await aiosqlite.connect(self.path)
        self.store = Store(self.db, history_turns=2)
        await self.store.initialize()

    async def asyncTearDown(self):
        await self.db.close()
        self.directory.cleanup()

    async def test_history_order_pruning_and_budget(self):
        for user, answer in [('a', 'A'), ('b', 'B'), ('c', 'C')]:
            await self.store.save_turn(user, answer)
        self.assertEqual([m['content'] for m in await self.store.history(100)], ['b', 'B', 'c', 'C'])
        self.assertEqual([m['content'] for m in await self.store.history(2)], ['c', 'C'])
        self.assertEqual(await self.store.history(1), [])
        async with self.db.execute('SELECT COUNT(*) FROM turns') as cur:
            self.assertEqual((await cur.fetchone())[0], 2)

    async def test_persistence_reset_and_memories(self):
        note_id = await self.store.remember('Используй aiogram 3')
        await self.store.save_turn('Привет', 'Привет, Даня')
        await self.db.close()
        self.db = await aiosqlite.connect(self.path)
        self.store = Store(self.db, 2)
        self.assertEqual(len(await self.store.history(100)), 2)
        await self.store.reset()
        self.assertEqual(await self.store.history(100), [])
        self.assertEqual(await self.store.memories(), [(note_id, 'Используй aiogram 3')])
        self.assertTrue(await self.store.forget(note_id))
        self.assertFalse(await self.store.forget(note_id))

    async def test_note_limits(self):
        with self.assertRaises(ValueError):
            await self.store.remember('x' * 1001)
        for i in range(30):
            await self.store.remember(str(i))
        with self.assertRaises(ValueError):
            await self.store.remember('overflow')


class AccessTests(unittest.IsolatedAsyncioTestCase):
    async def test_gate(self):
        for owner, user, chat, text, allowed, reply in [
            (42, 42, 'private', 'Привет', True, False),
            (42, 43, 'private', '/profile', False, True),
            (None, 42, 'private', 'Привет', False, True),
            (42, 42, 'group', 'Привет', False, False),
            (None, 42, 'private', '/id', False, True),
        ]:
            with self.subTest(owner=owner, user=user, chat=chat, text=text):
                event = SimpleNamespace(chat=SimpleNamespace(type=chat), from_user=SimpleNamespace(id=user), text=text, answer=AsyncMock())
                handler = AsyncMock()
                await OwnerOnly(owner)(handler, event, {})
                self.assertEqual(handler.await_count, int(allowed))
                self.assertEqual(event.answer.await_count, int(reply))


class LLMTests(unittest.IsolatedAsyncioTestCase):
    def config(self):
        return Config('fake-telegram', 'fake-api', 42, 'https://plusvibeapi.ru/v1',
                      'deepseek/deepseek-v4-flash-0731', 120, 4096, 20, 40000)

    async def test_request_and_truncation(self):
        def handle(request):
            self.assertEqual(str(request.url), 'https://plusvibeapi.ru/v1/chat/completions')
            self.assertEqual(request.headers['Authorization'], 'Bearer fake-api')
            body = json.loads(request.content)
            self.assertEqual(body['model'], self.config().model)
            self.assertEqual(body['messages'][0]['content'], 'test')
            self.assertFalse(body['stream'])
            return httpx.Response(200, json={'choices': [{'message': {'content': 'hello'}, 'finish_reason': 'length'}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            answer = await LLM(self.config(), client).answer([{'role': 'user', 'content': 'test'}])
        self.assertTrue(answer.startswith('hello'))
        self.assertIn('обрезан', answer)

    async def test_bad_responses_hide_provider_body(self):
        for status, payload in [(401, {'secret': 'sensitive'}), (429, {}), (500, {}), (200, {}), (200, {'choices': []}), (200, {'choices': [{'message': {'content': None}}]})]:
            with self.subTest(status=status, payload=payload):
                async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(status, json=payload))) as client:
                    with self.assertRaises(LLMError) as error:
                        await LLM(self.config(), client).answer([])
                    self.assertNotIn('sensitive', str(error.exception))

    async def test_timeout(self):
        def handle(request):
            raise httpx.ReadTimeout('secret', request=request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            with self.assertRaisesRegex(LLMError, 'не ответил вовремя'):
                await LLM(self.config(), client).answer([])


class TextTests(unittest.TestCase):
    def test_unicode_roundtrip(self):
        text = ('Привет 🖤\n```python\nprint(42)\n```\n' * 1000)
        chunks = list(split_text(text))
        self.assertEqual(''.join(chunks), text)
        self.assertTrue(all(len(chunk.encode('utf-16-le')) // 2 <= 3500 for chunk in chunks))


if __name__ == '__main__':
    unittest.main()
