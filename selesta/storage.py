import aiosqlite


class Store:
    def __init__(self, db: aiosqlite.Connection, history_turns: int):
        self.db = db
        self.history_turns = history_turns

    async def initialize(self):
        await self.db.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY, user_text TEXT NOT NULL, answer TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL
            );
        ''')
        await self.db.commit()

    async def history(self, char_budget: int):
        async with self.db.execute(
            'SELECT user_text, answer FROM turns ORDER BY id DESC LIMIT ?',
            (self.history_turns,),
        ) as cursor:
            rows = await cursor.fetchall()
        selected = []
        for user, answer in rows:
            size = len(user) + len(answer)
            if size > char_budget:
                break
            selected.append((user, answer))
            char_budget -= size
        return [
            {'role': role, 'content': text}
            for user, answer in reversed(selected)
            for role, text in [('user', user), ('assistant', answer)]
        ]

    async def save_turn(self, user: str, answer: str):
        await self.db.execute('INSERT INTO turns(user_text, answer) VALUES (?, ?)', (user, answer))
        await self.db.execute(
            'DELETE FROM turns WHERE id NOT IN (SELECT id FROM turns ORDER BY id DESC LIMIT ?)',
            (self.history_turns,),
        )
        await self.db.commit()

    async def reset(self):
        await self.db.execute('DELETE FROM turns')
        await self.db.commit()

    async def memories(self):
        async with self.db.execute('SELECT id, text FROM memories ORDER BY id') as cursor:
            return await cursor.fetchall()

    async def remember(self, text: str):
        if len(text) > 1000:
            raise ValueError('Заметка должна быть не длиннее 1000 символов.')
        if len(await self.memories()) >= 30:
            raise ValueError('Уже сохранено 30 заметок. Удали ненужные через /forget ID.')
        cursor = await self.db.execute('INSERT INTO memories(text) VALUES (?)', (text,))
        await self.db.commit()
        return cursor.lastrowid

    async def forget(self, memory_id: int):
        cursor = await self.db.execute('DELETE FROM memories WHERE id = ?', (memory_id,))
        await self.db.commit()
        return cursor.rowcount > 0
