"""Display actual completion logs without matching concurrent API requests."""
import time
import unittest

from aiohttp.test_utils import make_mocked_request
from test_dashboard import database, insert, trace


LINES = [
    'serve_api: mtp 34 tok in 2.31s = 14.70 t/s | 4 rounds, commit 2.00/round | prompt 423, prefill 0.49s (423 new) | detok 20us/tok | 26 tok with the head off | pool 673280/786432 86% | think on, closed at 1 by answer room',
    'serve_api: mtp 163 tok in 4.64s = 35.13 t/s | 92 rounds, commit 1.77/round | prompt 37219, prefill 23.43s = 1589 t/s | detok 24us/tok | pool 673280/786432 86% | think on | closest earlier: 213 tokens shared with a request 284s ago, then its prompt differs',
    'serve_api: mtp 162 tok in 5.23s = 30.96 t/s | 83 rounds, commit 1.94/round | prompt 39520 (38784 cached, 98.1%), prefill 0.76s (736 new) | detok 22us/tok | pool 774144/786432 98% | think on | disk: restored 38784 rows in 380 ms',
    'serve_api: mtp 108 tok in 4.01s = 26.93 t/s | 62 rounds, commit 1.73/round | prompt 20348 (20288 cached, 99.7%), prefill 0.61s (60 new) | detok 24us/tok | pool 764928/786432 97% | think on | disk: restored 20288 rows in 171 ms',
    'serve_api: mtp 108 tok in 3.63s = 29.73 t/s | 62 rounds, commit 1.73/round | prompt 20348 (20348 cached, 100.0%), prefill 0.00s | detok 23us/tok | pool 764928/786432 97% | think on',
    'serve_api: mtp 54 tok in 1.41s = 38.30 t/s | 28 rounds, commit 1.89/round | prompt 40527 (39872 cached, 98.4%), prefill 0.73s (655 new) | detok 23us/tok | pool 672768/786432 86% | think on | disk: restored 39872 rows in 233 ms',
]


class EngineHistoryTests(unittest.TestCase):
    def setUp(self):
        self.dashboard = database()
        self.addCleanup(self.dashboard.db.close)
        self.now = float(int(time.time()))

    def store(self, line, cursor='one', at=None):
        self.dashboard._store_engine_line({'MESSAGE': line,
            '_SYSTEMD_USER_UNIT': 'halogen-official.service', '__CURSOR': cursor,
            '__REALTIME_TIMESTAMP': str(int((at or self.now) * 1_000_000))})

    def test_real_completion_lines_keep_exact_logged_speed_time_and_cache(self):
        for i, line in enumerate(LINES):
            self.store(line, str(i), self.now + i)
        history = self.dashboard._engine_history(self.now - 1, self.now + 10)
        self.assertEqual((history['source'], history['total']), ('systemd_journal', 6))
        rows = list(reversed(history['items']))
        self.assertEqual([r['tps'] for r in rows], [14.70, 35.13, 30.96, 26.93, 29.73, 38.30])
        self.assertEqual([r['generation_s'] for r in rows], [2.31, 4.64, 5.23, 4.01, 3.63, 1.41])
        self.assertEqual((rows[1]['output_tokens'], rows[1]['prefill_s']), (163, 23.43))
        self.assertEqual((rows[2]['cached_tokens'], rows[2]['new_tokens']), (38784, 736))
        self.assertEqual(rows[0]['closed_reason'], 'closed at 1 by answer room')
        self.assertNotAlmostEqual(rows[2]['tps'], 162 / 5.23, places=3)
        self.assertEqual(self.dashboard._history(0, self.now + 10)['total'], 0)

    def test_duplicate_logs_and_progress_do_not_create_extra_completions(self):
        self.store(LINES[0])
        self.store(LINES[0])
        self.store('flash_serve: req 68 generated 3762 tokens, 60.1 s', 'progress')
        self.store('INFO: POST /v1/messages?beta=true HTTP/1.1 200 OK', 'access')
        self.assertEqual(self.dashboard._engine_history(0, self.now + 1)['total'], 1)

    def test_independent_pagination_and_reporting_range(self):
        for i in range(25):
            self.store(LINES[0], str(i), self.now + i)
        for i in range(12):
            insert(self.dashboard, completed_at=self.now)
        request = make_mocked_request('GET', f'/dashboard/api/analytics?period=custom&from={self.now-1}&to={self.now+100}&page=2&engine_page=3')
        # Query a fixed completed interval so future fixture records are in range.
        self.dashboard._resolve_range = lambda _: ('custom', self.now - 1, self.now + 100)
        result = self.dashboard.analytics(request)
        self.assertEqual((result['history']['page'], len(result['history']['items'])), (2, 2))
        self.assertEqual((result['engine_history']['page'], len(result['engine_history']['items'])), (3, 5))
        self.assertEqual(self.dashboard._engine_history(self.now + 10, self.now + 12)['total'], 3)
        self.assertEqual(self.dashboard._engine_history(0, self.now + 100, 999)['page'], 3)
        self.assertEqual(self.dashboard._engine_history(0, self.now + 100, page_size=999)['page_size'], 100)

    def test_same_tokens_and_timestamps_never_attach_engine_values_to_api_rows(self):
        insert(self.dashboard, completed_at=self.now, output_tokens=108, tps=None)
        insert(self.dashboard, completed_at=self.now, output_tokens=108, tps=None)
        self.store(LINES[3], 'a')
        self.store(LINES[4], 'b')
        self.assertTrue(all(row['tps'] is None for row in self.dashboard._history(0, self.now + 1)['items']))
        self.assertEqual({row['tps'] for row in self.dashboard._engine_history(0, self.now + 1)['items']}, {26.93, 29.73})

    def test_missing_fields_remain_unknown_and_empty_history_has_valid_paging(self):
        empty = self.dashboard._engine_history(0, self.now)
        self.assertEqual((empty['total'], empty['page'], empty['pages']), (0, 1, 1))
        self.store('serve_api: mtp 10 tok in 0.0s = n/a t/s')
        row = self.dashboard._engine_history(0, self.now + 1)['items'][0]
        self.assertEqual(row['generation_s'], 0)
        self.assertIsNone(row['tps'])
        self.assertIsNone(row['prefill_s'])


class NoObservedRatesTests(unittest.IsolatedAsyncioTestCase):
    async def test_old_observation_columns_are_ignored_without_deleting_them(self):
        dashboard = database()
        self.addCleanup(dashboard.db.close)
        for field in ('first_output_ms', 'output_stream_ms', 'observed_tps'):
            dashboard.db.execute(f'ALTER TABLE requests ADD COLUMN {field} REAL')
        insert(dashboard, endpoint='/v1/messages', tps=None, observed_tps=999)
        dashboard._initialize_schema()
        row = dashboard._history(0, 20000)['items'][0]
        self.assertNotIn('observed_tps', row)
        self.assertIsNone(row['tps'])
        self.assertEqual(dashboard.db.execute('SELECT observed_tps FROM requests').fetchone()[0], 999)

    async def test_complete_messages_stream_does_not_compute_generation_speed(self):
        dashboard = database()
        self.addCleanup(dashboard.db.close)
        item = trace('/v1/messages', True)
        item.observe(b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"OK"}}\n\n', 'text/event-stream')
        item.observe(b'data: {"type":"message_delta","usage":{"input_tokens":58,"output_tokens":29}}\n\ndata: {"type":"message_stop"}\n\n', 'text/event-stream')
        await dashboard.finish_request(item, 200)
        row = dashboard.db.execute('SELECT tps,input_tokens,output_tokens FROM requests').fetchone()
        self.assertEqual(tuple(row), (None, 58, 29))
