import asyncio
import logging

from .bot import main

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

try:
    asyncio.run(main())
except (KeyboardInterrupt, SystemExit):
    pass
except Exception as exc:
    # Never print raw startup exceptions that might embed tokens or request URLs.
    if isinstance(exc, ValueError) and type(exc) is ValueError:
        logging.error('%s', exc)
    else:
        logging.error('Запуск остановлен: %s. Проверь .env и подключение.', type(exc).__name__)
    raise SystemExit(1) from None
