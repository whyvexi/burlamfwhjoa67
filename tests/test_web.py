import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from selesta.config import Config
from selesta.llm import LLM, LLMError
from selesta.web import MAX_TEXT, PublicResolver, WebError, WebTools, extract_text, validate_url


class WebTests(unittest.IsolatedAsyncioTestCase):
    def test_reject_private_and_non_web_urls(self):
        for url in ['file:///etc/passwd', 'http://localhost', 'http://127.0.0.1',
                    'http://10.0.0.1', 'http://169.254.169.254', 'http://[::1]',
                    'http://[::ffff:127.0.0.1]', 'https://example.com:8080',
                    'https://user:pass@example.com', 'http://server.local']:
            with self.subTest(url=url), self.assertRaises(WebError):
                validate_url(url)
        self.assertEqual(str(validate_url('https://example.com/docs#part')), 'https://example.com/docs')

    async def test_dns_checks_actual_connection_addresses(self):
        resolver = PublicResolver()
        try:
            with patch('selesta.web.ThreadedResolver.resolve', new=AsyncMock(return_value=[{'host': '127.0.0.1'}])):
                with self.assertRaises(WebError):
                    await resolver.resolve('external.example')
        finally:
            await resolver.close()

    def test_html_extraction_and_limit(self):
        raw = b'<html><title>Docs</title><script>secret</script><nav>menu</nav><main><h1>Hello</h1><pre>print(42)</pre></main></html>'
        result = extract_text(raw, 'text/html')
        self.assertEqual(result['title'], 'Docs')
        self.assertIn('print(42)', result['text'])
        self.assertNotIn('secret', result['text'])
        self.assertNotIn('menu', result['text'])
        result = extract_text(b'x' * (MAX_TEXT + 1), 'text/plain')
        self.assertEqual(len(result['text']), MAX_TEXT)
        self.assertTrue(result['truncated'])

    async def test_invalid_calls_never_search(self):
        web = WebTools()
        web.search = AsyncMock()
        for name, args in [('shell', '{}'), ('web_search', 'broken'),
                           ('web_search', '{"query": 3}'), ('web_search', '{}'),
                           ('web_search', '{"query":"ok","extra":1}')]:
            self.assertIn('error', await web.execute(name, args))
        web.search.assert_not_awaited()

    async def test_search_results_and_errors(self):
        with patch('selesta.web.DDGS') as ddgs:
            ddgs.return_value.text.return_value = [{'href': 'https://example.com', 'title': 'Test', 'body': 'Text'},
                                                 {'href': 'http://127.0.0.1', 'body': 'private'}]
            result = await WebTools().execute('web_search', '{"query":"test"}')
            self.assertEqual(len(result['results']), 1)
            self.assertTrue(result['untrusted_web_data'])
            ddgs.return_value.text.side_effect = RuntimeError('secret')
            result = await WebTools().execute('web_search', '{"query":"test"}')
            self.assertIn('error', result)
            self.assertNotIn('secret', result['error'])

    async def test_redirect_cannot_reach_private_host(self):
        class Response:
            status = 302
            headers = {'Location': 'http://127.0.0.1/secrets'}
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
        with patch('selesta.web.aiohttp.ClientSession.get', return_value=Response()) as get:
            result = await WebTools().execute('read_url', '{"url":"https://example.com"}')
            self.assertIn('error', result)
            self.assertEqual(get.call_count, 1)


class AgentTests(unittest.IsolatedAsyncioTestCase):
    def config(self):
        return Config('fake', 'fake', 42, 'https://plusvibeapi.ru/v1', 'deepseek/deepseek-v4-flash-0731', 120, 4096, 20, 40000)

    def tool_response(self, count=1):
        return {'choices': [{'message': {'content': None, 'reasoning_content': 'reason', 'tool_calls': [
            {'id': f'call_{i}', 'type': 'function', 'function': {'name': 'web_search', 'arguments': '{"query":"test"}'}}
            for i in range(count)
        ]}}]}

    async def test_tool_roundtrip_with_sources(self):
        requests = []
        def handle(request):
            body = json.loads(request.content)
            requests.append(body)
            if len(requests) == 1:
                return httpx.Response(200, json=self.tool_response())
            self.assertEqual(body['messages'][-1]['role'], 'tool')
            self.assertEqual(body['messages'][-1]['tool_call_id'], 'call_0')
            self.assertEqual(body['messages'][-2]['reasoning_content'], 'reason')
            return httpx.Response(200, json={'choices': [{'message': {'content': 'Готово'}}]})
        web = AsyncMock()
        web.execute.return_value = {'results': [{'url': 'https://example.com'}]}
        messages = [{'role': 'user', 'content': 'Найди test'}]
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            answer = await LLM(self.config(), client, web).answer(messages)
        self.assertIn('https://example.com', answer)
        self.assertEqual(len(messages), 1)
        self.assertEqual(len(requests), 2)
        web.execute.assert_awaited_once_with('web_search', '{"query":"test"}')

    async def test_tool_budget_forces_final_answer(self):
        requests = []
        def handle(request):
            body = json.loads(request.content)
            requests.append(body)
            if body['tool_choice'] == 'none':
                return httpx.Response(200, json={'choices': [{'message': {'content': 'Итог'}}]})
            return httpx.Response(200, json=self.tool_response(count=4))
        web = AsyncMock()
        web.execute.return_value = {'error': 'blocked'}
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            answer = await LLM(self.config(), client, web).answer([])
        self.assertEqual(answer, 'Итог')
        self.assertEqual(web.execute.await_count, 6)
        self.assertEqual(requests[-1]['tool_choice'], 'none')
        self.assertEqual(len(requests), 3)

    async def test_explicit_read_runs_before_model(self):
        web = AsyncMock()
        web.execute.return_value = {'url': 'https://example.com', 'text': 'Example'}
        def handle(request):
            web.execute.assert_awaited_once()
            body = json.loads(request.content)
            self.assertEqual(body['messages'][-1]['role'], 'tool')
            return httpx.Response(200, json={'choices': [{'message': {'content': 'https://example.com'}}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            await LLM(self.config(), client, web).answer([], initial_tool=('read_url', {'url': 'https://example.com'}))

    async def test_malformed_tool_call(self):
        payload = self.tool_response()
        payload['choices'][0]['message']['tool_calls'][0]['function'] = None
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))) as client:
            with self.assertRaises(LLMError):
                await LLM(self.config(), client).answer([])
