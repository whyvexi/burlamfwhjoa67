"""Read-only public web tools. No LLM credentials are passed to websites."""
import asyncio
import ipaddress
import json
import socket
import ssl

import aiohttp
import certifi
from aiohttp.resolver import ThreadedResolver
from bs4 import BeautifulSoup
from ddgs import DDGS
from yarl import URL

MAX_BYTES = 2_000_000
MAX_TEXT = 14000

TOOLS = [
    {'type': 'function', 'function': {
        'name': 'web_search',
        'description': 'Поиск актуальной информации в интернете. Возвращает ссылки и краткие выдержки. Для подробностей прочитай страницу через read_url.',
        'parameters': {'type': 'object', 'properties': {
            'query': {'type': 'string', 'description': 'Короткий поисковый запрос без секретов и личного профиля'},
        }, 'required': ['query'], 'additionalProperties': False},
    }},
    {'type': 'function', 'function': {
        'name': 'read_url',
        'description': 'Прочитать публичную HTTP(S) страницу: HTML или обычный текст. Не поддерживает PDF, авторизацию и выполнение JavaScript.',
        'parameters': {'type': 'object', 'properties': {
            'url': {'type': 'string', 'description': 'Полный URL страницы'},
        }, 'required': ['url'], 'additionalProperties': False},
    }},
]


class WebError(Exception):
    pass


def public_ip(value: str):
    address = ipaddress.ip_address(value.split('%')[0])
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    if not address.is_global or address.is_multicast:
        raise WebError('Доступ к локальным и служебным адресам запрещён.')


def validate_url(value: str) -> URL:
    try:
        url = URL(value)
        if url.scheme not in {'http', 'https'} or not url.host or url.user is not None:
            raise ValueError
        if url.port not in {80, 443} or '%' in url.host:
            raise ValueError
        host = url.host.rstrip('.').lower()
        if host == 'localhost' or host.endswith(('.localhost', '.local', '.internal')):
            raise WebError('Доступ к локальным адресам запрещён.')
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            public_ip(host)
        return url.with_fragment(None)
    except (ValueError, TypeError, UnicodeError):
        raise WebError('Нужен публичный HTTP(S) URL без логина, со стандартным портом.') from None


class PublicResolver(ThreadedResolver):
    async def resolve(self, host, port=0, family=socket.AF_INET):
        records = await super().resolve(host, port, family)
        # Validate the exact addresses the connector will use (no separate DNS preflight).
        for record in records:
            public_ip(record['host'])
        return records


def extract_text(raw: bytes, content_type: str, encoding: str = 'utf-8'):
    try:
        text = raw.decode(encoding, errors='replace')
    except LookupError:
        text = raw.decode('utf-8', errors='replace')
    title = ''
    if content_type in {'text/html', 'application/xhtml+xml'}:
        soup = BeautifulSoup(text, 'html.parser')
        title = soup.title.get_text(' ', strip=True) if soup.title else ''
        for tag in soup(['script', 'style', 'noscript', 'svg', 'nav', 'footer', 'header', 'form']):
            tag.decompose()
        main = soup.find('main') or soup.find('article') or soup.body or soup
        text = main.get_text('\n', strip=True)
    if not text.strip():
        raise WebError('На странице нет доступного текста; возможно, нужен JavaScript.')
    return {'title': title[:300], 'text': text[:MAX_TEXT], 'truncated': len(text) > MAX_TEXT}


class WebTools:
    async def search(self, query: str):
        def run():
            return DDGS(timeout=12).text(query, max_results=5, backend='duckduckgo')

        rows = await asyncio.wait_for(asyncio.to_thread(run), timeout=20)
        results = []
        for row in rows[:5]:
            try:
                url = str(validate_url(row.get('href', '')))
            except WebError:
                continue
            results.append({'title': str(row.get('title', ''))[:300], 'url': url,
                            'snippet': str(row.get('body', ''))[:1500]})
        if not results:
            raise WebError('Поиск не дал результатов. Попробуй другой запрос или конкретную ссылку.')
        return {'results': results}

    async def read(self, value: str):
        url = validate_url(value)
        connector = aiohttp.TCPConnector(
            resolver=PublicResolver(), use_dns_cache=False,
            ssl=ssl.create_default_context(cafile=certifi.where()),
        )
        async with aiohttp.ClientSession(
            connector=connector, trust_env=False, cookie_jar=aiohttp.DummyCookieJar(),
            timeout=aiohttp.ClientTimeout(total=20),
            headers={'User-Agent': 'Selesta/1.0 (public-page reader)'},
        ) as session:
            async with asyncio.timeout(30):
                for _ in range(6):
                    async with session.get(url, allow_redirects=False) as response:
                        if response.status in {301, 302, 303, 307, 308}:
                            location = response.headers.get('Location')
                            if not location:
                                raise WebError('Сайт вернул перенаправление без адреса.')
                            url = validate_url(str(url.join(URL(location))))
                            continue
                        if response.status >= 400:
                            raise WebError(f'Сайт ответил HTTP {response.status}; текст не получен.')
                        content_type = response.content_type
                        if content_type not in {'text/html', 'application/xhtml+xml', 'text/plain', 'text/markdown'}:
                            raise WebError('Этот формат не поддерживается. Пришли HTML-страницу или текст.')
                        raw = bytearray()
                        async for chunk in response.content.iter_chunked(32768):
                            raw.extend(chunk)
                            if len(raw) > MAX_BYTES:
                                raise WebError('Страница больше 2 МБ; выбери более короткую страницу.')
                        return {'url': str(url), **extract_text(bytes(raw), content_type, response.charset or 'utf-8')}
                raise WebError('Слишком много перенаправлений.')

    async def execute(self, name: str, arguments: str):
        try:
            if not isinstance(arguments, str) or len(arguments) > 8000:
                raise WebError('Некорректные аргументы инструмента.')
            args = json.loads(arguments)
            field = {'web_search': 'query', 'read_url': 'url'}.get(name)
            if not field or not isinstance(args, dict) or set(args) != {field}:
                raise WebError('Неизвестный инструмент или некорректные аргументы.')
            value = args[field]
            if not isinstance(value, str) or not value.strip() or len(value) > (500 if field == 'query' else 4000):
                raise WebError('Аргумент пустой или слишком длинный.')
            result = await (self.search(value.strip()) if field == 'query' else self.read(value.strip()))
            return {'untrusted_web_data': True, **result}
        except WebError as exc:
            return {'error': str(exc)}
        except json.JSONDecodeError:
            return {'error': 'Аргументы инструмента должны быть JSON.'}
        except (TimeoutError, asyncio.TimeoutError):
            return {'error': 'Истекло время ожидания сайта или поиска.'}
        except Exception:
            # Provider errors may include URLs or private request data.
            return {'error': 'Не удалось получить данные. Возможно, сайт или поиск блокирует запросы. Попробуй другую ссылку.'}
