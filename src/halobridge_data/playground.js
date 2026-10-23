'use strict';
// The playground sends only explicit requests to documented routes on this router.
(function (root) {
  const gpu = ['qwen3.8-flash'];
  const entry = (id, path, options = {}) => ({id, path, method: 'POST', group: 'gpu', models: gpu, ...options});
  const ENDPOINTS = [
    entry('chat', '/v1/chat/completions'),
    entry('vision', '/v1/chat/completions', {file: 'vision'}),
    entry('tools', '/v1/chat/completions'),
    entry('structured', '/v1/chat/completions'),
    entry('guard', '/v1/chat/completions'),
    entry('responses', '/v1/responses'),
    entry('completions', '/v1/completions'),
    entry('messages', '/v1/messages'),
    entry('count_tokens', '/v1/messages/count_tokens'),
    entry('decision', '/v1/chat/completions', {group: 'npu', models: ['decider-4b', 'decider-0.8b']}),
    entry('systemone', '/v1/systemone', {group: 'npu', models: ['d3-lite', 'd3-nano', 'decider-4b', 'decider-0.8b']}),
    entry('systemone_image', '/v1/systemone', {group: 'npu', models: ['d3-lite', 'd3-nano'], file: 'systemone_image'}),
    entry('embedding', '/v1/embeddings', {group: 'npu', models: ['qwen3-embedding-0.6b', 'qwen3-embedding-4b', 'embeddinggemma-2', 'wemm-embedding-4b']}),
    entry('embedding_image', '/v1/embeddings', {group: 'npu', models: ['embeddinggemma-2', 'wemm-embedding-4b'], file: 'embedding_image'}),
    entry('embedding_audio', '/v1/embeddings', {group: 'npu', models: ['embeddinggemma-2'], file: 'embedding_audio'}),
    entry('rerank', '/v1/rerank', {group: 'npu', models: ['qwen3-reranker-0.6b', 'qwen3-reranker-4b']}),
    entry('moderations', '/v1/moderations', {group: 'npu', models: ['qwen3guard-gen-0.6b']}),
    entry('npu_text', '/v1/chat/completions', {group: 'npu', models: ['qwen3.5-2b']}),
    entry('classify', '/classify', {group: 'npu', models: ['my-classifier']}),
    entry('images', '/v1/images/generations', {group: 'npu', models: ['flux2-klein-4b'], image: true}),
    entry('edits', '/v1/images/edits', {group: 'npu', models: ['flux2-klein-4b'], file: 'edits', multipart: true, image: true}),
    entry('variations', '/v1/images/variations', {group: 'npu', models: ['flux2-klein-4b'], file: 'variations', multipart: true}),
    entry('transcriptions', '/v1/audio/transcriptions', {group: 'npu', models: ['qwen3-asr-1.7b', 'qwen3-asr-0.6b'], file: 'audio', multipart: true}),
    entry('speech', '/v1/audio/speech', {group: 'npu', models: ['qwen3-tts-1.7b-customvoice', 'qwen3-tts-0.6b-customvoice']}),
    ...['models', 'health', 'metrics', 'cache', 'props', 'slots', 'handles', 'router'].map(id => entry(id,
      {models: '/v1/models', handles: '/halogen/handles', router: '/router/status'}[id] || '/' + id,
      {method: 'GET', group: 'inspect', models: []})),
    entry('tokenize', '/tokenize', {group: 'inspect', models: []}),
    entry('detokenize', '/detokenize', {group: 'inspect', models: []}),
    entry('release', '/halogen/handles/', {method: 'DELETE', group: 'inspect', models: []}),
  ];
  const STREAM_IDS = new Set(['chat', 'tools', 'structured', 'guard', 'responses', 'completions', 'messages', 'npu_text', 'speech']);
  function sample(e, model) {
    const chat = {model, messages: [{role: 'user', content: 'Explain why the sky is blue in three sentences.'}], max_tokens: 512, enable_thinking: false};
    const img = {model, prompt: 'A small red cabin beside a lake, watercolor illustration', size: '512x512', n: 1, response_format: 'b64_json'};
    switch (e.id) {
      case 'vision': return {...chat, messages: [{role: 'user', content: 'Describe this image.'}]};
      case 'tools': return {...chat, messages: [{role: 'user', content: 'What is the weather in Berlin?'}], tools: [{type: 'function', function: {name: 'get_weather', description: 'Look up the weather for a city', strict: true, parameters: {type: 'object', properties: {city: {type: 'string'}}, required: ['city'], additionalProperties: false}}}], parallel_tool_calls: false};
      case 'structured': return {...chat, messages: [{role: 'user', content: 'Classify this review: The product arrived quickly and works well.'}], response_format: {type: 'json_schema', json_schema: {name: 'sentiment', strict: true, schema: {type: 'object', properties: {sentiment: {type: 'string', enum: ['positive', 'negative', 'neutral']}}, required: ['sentiment'], additionalProperties: false}}}};
      case 'guard': return {...chat, guard: true, stream: true};
      case 'responses': return {model, input: 'Explain why the sky is blue in three sentences.', max_output_tokens: 512};
      case 'completions': return {model, prompt: 'Three practical uses for a local AI server:\n1.', max_tokens: 256};
      case 'messages': return {model, messages: chat.messages, max_tokens: 512};
      case 'count_tokens': return {model, messages: chat.messages};
      case 'decision': return {model, messages: [{role: 'user', content: 'WIN a FREE iPhone! Click here NOW to claim your prize!'}], response_format: {type: 'json_schema', json_schema: {name: 'spam', description: 'Is this message spam?', schema: {enum: ['no', 'yes']}}}, logprobs: true, top_logprobs: 2};
      case 'systemone': return {model, state: 'The customer was charged twice and asks for a refund.', questions: {team: {type: 'choice', instructions: 'Which team should handle this?', criteria: {billing: 'Payment problems', technical: 'Software problems', other: null}}}};
      case 'systemone_image': return {model, state: 'Look at the supplied image.', questions: {scene: {type: 'choice', instructions: 'Where was this picture taken?', criteria: {indoors: 'Inside a building', outdoors: 'Outside a building', uncertain: null}}}};
      case 'embedding': return {model, input: ['How do I enable the NPU?'], encoding_format: 'float'};
      case 'embedding_image': case 'embedding_audio': return {model, input: []};
      case 'rerank': return {model, query: 'How do I enable the NPU?', documents: ['Install the NPU driver and XRT plugin.', 'The weather is sunny today.'], top_n: 2};
      case 'moderations': return {model, input: 'Explain how to store passwords safely.'};
      case 'classify': return {model, input: ['Win a free prize now', 'Lunch at noon?']};
      case 'npu_text': return {...chat, messages: [{role: 'user', content: 'Summarize: A local server can process data without uploading it to a cloud service.'}]};
      case 'images': return img;
      case 'edits': return {...img, prompt: 'Turn image 1 into a pencil sketch.'};
      case 'variations': return {model, n: 1, response_format: 'b64_json'};
      case 'transcriptions': return {model, response_format: 'json'};
      case 'speech': return {model, input: 'Hello! This is a speech test running locally on the NPU.', voice: 'ryan', language: 'auto', response_format: 'wav'};
      case 'tokenize': return {content: 'Hello from Halobridge', add_special: false};
      case 'detokenize': return {tokens: [9707]};
      default: return e.method === 'GET' || e.method === 'DELETE' ? null : chat;
    }
  }
  function textValue(e, body) {
    if (!body) return '';
    if (e.id.startsWith('systemone')) return typeof body.state === 'string' ? body.state : JSON.stringify(body.state);
    if (e.id === 'rerank') return body.query || '';
    if (e.id === 'tokenize') return body.content || '';
    if (e.id === 'detokenize') return JSON.stringify(body.tokens || []);
    if (body.messages) { const value = body.messages[0]?.content; return typeof value === 'string' ? value : (value || []).find(x => x.type === 'text')?.text || ''; }
    const value = body.prompt ?? body.input;
    return Array.isArray(value) ? value.filter(x => typeof x === 'string').join('\n') : typeof value === 'string' ? value : '';
  }
  function setText(e, body, text) {
    if (body.messages) body.messages[0].content = text;
    else if (e.id.startsWith('systemone')) body.state = text;
    else if (e.id === 'rerank') body.query = text;
    else if (e.id === 'tokenize') body.content = text;
    else if (e.id === 'detokenize') { body.tokens = JSON.parse(text); if (!Array.isArray(body.tokens) || body.tokens.some(x => !Number.isInteger(x) || x < 0)) throw new Error('pg_invalid_tokens'); }
    else if ('prompt' in body) body.prompt = text;
    else body.input = Array.isArray(body.input) ? text.split('\n').filter(Boolean) : text;
    return body;
  }
  function pcmWav(pcm) {
    if (pcm.byteLength % 2) throw new Error('pg_invalid_pcm');
    const output = new Uint8Array(44 + pcm.byteLength), view = new DataView(output.buffer);
    function str(offset, s) { for (let i = 0; i < s.length; i++) output[offset + i] = s.charCodeAt(i); }
    str(0, 'RIFF'); view.setUint32(4, pcm.byteLength + 36, true); str(8, 'WAVEfmt ');
    view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, 24000, true); view.setUint32(28, 48000, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    str(36, 'data'); view.setUint32(40, pcm.byteLength, true); output.set(pcm, 44); return output;
  }
  function sseParser(emit) {
    let buffer = '';
    return {push(text, final = false) {
      buffer += text;
      let offset;
      while ((offset = buffer.indexOf('\n')) >= 0) { const line = buffer.slice(0, offset).replace(/\r$/, ''); buffer = buffer.slice(offset + 1); consume(line); }
      if (buffer.length > 2 * 1024 * 1024) throw new Error('pg_response_limit');
      if (final && buffer) { consume(buffer); buffer = ''; }
    }};
    function consume(line) {
      if (!line.startsWith('data:')) return;
      const value = line.slice(5).trim(); if (!value || value === '[DONE]') return;
      try { emit(JSON.parse(value)); } catch (error) { if (!(error instanceof SyntaxError)) throw error; }
    }
  }
  function answer(value) {
    if (value.error) return typeof value.error === 'string' ? value.error : value.error.message || JSON.stringify(value.error);
    if (value.choices) return value.choices.map(c => {
      const m = c.delta || c.message || {};
      return (m.reasoning_content || '') + (m.content || '') + (c.text || '') + (m.tool_calls ? '\n' + JSON.stringify(m.tool_calls, null, 2) : '');
    }).join('');
    if (value.type === 'response.output_text.delta' || value.type === 'response.reasoning_summary_text.delta') return value.delta || '';
    if (value.type === 'content_block_delta') return value.delta?.text || value.delta?.thinking || value.delta?.partial_json || '';
    if (Array.isArray(value.content)) return value.content.map(x => x.text || x.thinking || '').join('\n');
    if (Array.isArray(value.output)) return value.output.map(x => (x.content || []).map(c => c.text || '').join('')).join('\n');
    return value.text || value.output_text || '';
  }
  function extraHeaders(text, key) {
    const values = JSON.parse(text || '{}');
    if (!values || Array.isArray(values) || typeof values !== 'object') throw new Error('pg_invalid_headers');
    const headers = {'X-Halogen-Client': 'Halobridge API Playground'};
    for (const [name, value] of Object.entries(values)) {
      if (!/^(x-halogen-[a-z0-9-]+|anthropic-version|anthropic-beta)$/i.test(name) || typeof value !== 'string' || /[\r\n]/.test(value)) throw new Error('pg_invalid_headers');
      headers[name] = value;
    }
    if (key) headers.Authorization = 'Bearer ' + key;
    return headers;
  }
  async function preparePayload(e, b, files, dataURL) {
    if (e.file) {
      if (!files.length || files.length > (e.file === 'edits' ? 4 : 1)) throw new Error('pg_file_count');
      for (const file of files) {
        if (file.size > 25 * 1024 * 1024) throw new Error('pg_file_size');
        if ((e.file.includes('audio') || e.file === 'audio') && !/\.wav$/i.test(file.name)) throw new Error('pg_wav_hint');
        if (!e.file.includes('audio') && e.file !== 'audio' && !/\.(png|jpe?g|webp|gif|bmp)$/i.test(file.name)) throw new Error('pg_image_hint');
      }
    }
    if (e.multipart) {
      const form = new FormData();
      for (const [name, value] of Object.entries(b)) if (value !== null && value !== undefined) form.append(name, typeof value === 'object' ? JSON.stringify(value) : String(value));
      for (const file of files) form.append(e.file === 'audio' ? 'file' : 'image', file, file.name);
      return form;
    }
    if (e.file === 'vision') {
      const text = textValue(e, b); b.messages[0].content = [{type: 'text', text}, {type: 'image_url', image_url: {url: await dataURL(files[0])}}];
      } else if (e.file === 'systemone_image') b.image = {url: await dataURL(files[0])};
      else if (e.file === 'embedding_image') b.input = [{type: 'image_url', image_url: {url: await dataURL(files[0])}}];
    else if (e.file === 'embedding_audio') b.input = [{type: 'input_audio', input_audio: {data: (await dataURL(files[0])).split(',')[1], format: 'wav'}}];
    return b ? JSON.stringify(b) : undefined;
  }
  function create({t, apiFetch, getLive, seconds, bytes}) {
    const $ = id => document.getElementById(id);
    let selected = ENDPOINTS[0], models = [], loaded = false, loading = false, running = false, controller, timer;
    let urls = [], received = 0, started = 0, raw = '', output = '', resultState = 'idle', statusCode = '', editError = '', inputError = '';
    const MAX_RESPONSE = 128 * 1024 * 1024, MAX_TEXT = 2 * 1024 * 1024;
    function body() { const b = JSON.parse($('pgBody').value || 'null'); if (b !== null && (!b || Array.isArray(b) || typeof b !== 'object')) throw new Error('pg_invalid_json'); return b; }
    function save(b) { $('pgBody').value = b === null ? '' : JSON.stringify(b, null, 2); editError = ''; inputError = ''; }
    function requirement() {
      let key = selected.group === 'npu' ? 'pg_need_npu' : selected.group === 'gpu' ? 'pg_need_gpu' : 'pg_need_inspect';
      if (selected.id === 'guard') key = 'pg_need_guard';
      if (selected.id === 'speech') key = 'pg_need_speech';
      $('pgRequirement').textContent = t(key);
    }
    function translate() {
      const value = $('pgEndpoint').value;
      $('pgEndpoint').replaceChildren();
      for (const group of ['gpu', 'npu', 'inspect']) {
        const node = document.createElement('optgroup'); node.label = t('pg_group_' + group);
        for (const e of ENDPOINTS.filter(x => x.group === group)) { const o = document.createElement('option'); o.value = e.id; o.textContent = t('pg_example_' + e.id); node.append(o); }
        $('pgEndpoint').append(node);
      }
      $('pgEndpoint').value = value || selected.id;
      $('pgPurpose').textContent = t('pg_example_' + selected.id);
      requirement(); renderStatus();
      $('pgFileLabel').textContent = t(selected.file?.includes('audio') ? 'pg_audio_file' : 'pg_image_files');
      $('pgFileHint').textContent = t(selected.file === 'edits' ? 'pg_edit_files' : selected.file === 'audio' ? 'pg_wav_hint' : selected.file === 'embedding_audio' ? 'pg_embed_audio_hint' : 'pg_image_hint');
      if (loaded) modelStatus();
    }
    function modelStatus() { $('pgModelStatus').textContent = models.length ? t('pg_models_loaded', {n: models.length}) : t('pg_models_empty'); }
    function choose() {
      if (!selected.models.length) return '';
      const active = getLive()?.api?.active_model;
      if (selected.group === 'gpu') return active || models.find(x => !x.task)?.id || gpu[0];
      return selected.models.find(id => models.some(x => x.id === id)) || selected.models[0];
    }
    function reset() {
      selected = ENDPOINTS.find(x => x.id === $('pgEndpoint').value) || ENDPOINTS[0];
      $('pgModel').value = choose(); $('pgModel').disabled = !selected.models.length;
      $('pgRoute').textContent = selected.method + ' ' + selected.path + (selected.id === 'release' ? '<name>' : '');
      const b = sample(selected, $('pgModel').value); save(b);
      $('pgInput').value = textValue(selected, b);
      $('pgInputField').hidden = !b || ['variations', 'transcriptions', 'embedding_image', 'embedding_audio'].includes(selected.id);
      $('pgBodyField').hidden = !b; $('pgFiles').value = ''; $('pgConfirm').checked = false;
      $('pgFileField').hidden = !selected.file; $('pgFiles').multiple = selected.file === 'edits';
      $('pgFiles').accept = selected.file?.includes('audio') || selected.file === 'audio' ? '.wav,audio/wav,audio/x-wav' : 'image/png,image/jpeg,image/webp,image/gif,image/bmp';
      $('pgSpeechFields').hidden = selected.id !== 'speech'; $('pgVoice').value = 'ryan'; $('pgLanguage').value = 'auto'; $('pgInstructions').value = '';
      $('pgSizeField').hidden = !selected.image; $('pgSize').value = '512x512';
      $('pgStreamField').hidden = !STREAM_IDS.has(selected.id); $('pgStream').checked = !!b?.stream;
      $('pgHandleField').hidden = selected.id !== 'release'; $('pgConfirmField').hidden = selected.id !== 'release';
      translate();
    }
    function edit(fn) { try { const b = body(); if (b) { fn(b); save(b); } } catch (e) { inputError = e.message.startsWith('pg_') ? e.message : 'pg_invalid_json'; } }
    function renderStatus() {
      const status = resultState === 'failed' ? (statusCode ? ' · HTTP ' + statusCode : '') : statusCode;
      $('pgStatus').textContent = t('pg_' + resultState, {status}) + (resultState === 'failed' && editError ? ' ' + t(editError) : '');
      $('pgStatus').className = 'pg-status ' + (resultState === 'failed' ? 'failure' : '');
    }
    function clear() {
      for (const url of urls) URL.revokeObjectURL(url); urls = [];
      $('pgMedia').replaceChildren(); $('pgAnswer').textContent = ''; $('pgAnswer').hidden = true;
      $('pgRaw').textContent = ''; $('pgResponseHeaders').textContent = ''; raw = ''; output = ''; received = 0; started = 0;
      for (const id of ['pgTotal', 'pgFirstByte', 'pgReceived']) $(id).textContent = '–';
      resultState = 'idle'; statusCode = ''; editError = ''; renderStatus();
    }
    function busy(on) {
      running = on;
      for (const el of $('pgForm').querySelectorAll('input,textarea,select,button')) if (el.id !== 'pgStop') el.disabled = on;
      $('pgModel').disabled = on || !selected.models.length; $('pgEndpoint').disabled = on; $('pgRefresh').disabled = on || loading;
      $('pgClear').disabled = on; $('pgStop').hidden = !on;
    }
    function download(blob, filename, media) {
      const url = URL.createObjectURL(blob); urls.push(url);
      if (media) { const node = document.createElement(media); node.src = url; if (media === 'audio') node.controls = true; else node.alt = t('pg_generated_image'); $('pgMedia').append(node); }
      const a = document.createElement('a'); a.href = url; a.download = filename; a.textContent = t('pg_download') + ' · ' + filename; $('pgMedia').append(a);
    }
    function imageResults(value) {
      if (!Array.isArray(value.data)) return;
      let count = 0;
      for (const item of value.data.slice(0, 4)) {
        const encoded = typeof item.b64_json === 'string' ? item.b64_json : /^data:image\/png;base64,([A-Za-z0-9+/=\r\n]+)$/.exec(item.url || '')?.[1];
        if (!encoded) continue;
        const bin = atob(encoded); const data = Uint8Array.from(bin, c => c.charCodeAt(0));
        download(new Blob([data], {type: 'image/png'}), `halobridge-${++count}.png`, 'img');
      }
    }
    async function refreshModels() {
      if (loading || running) return; loading = true; $('pgRefresh').disabled = true;
      try {
        const response = await apiFetch('/v1/models', {cache: 'no-store', headers: extraHeaders('{}', $('pgKey').value)});
        if (!response.ok) throw new Error(String(response.status));
        const value = await response.json(); models = Array.isArray(value.data) ? value.data.filter(x => typeof x.id === 'string') : [];
        $('pgModels').replaceChildren(...models.map(m => { const o = document.createElement('option'); o.value = m.id; o.label = m.task || 'GPU'; return o; }));
        if (!loaded && !running && $('pgModel').value === selected.models[0]) { $('pgModel').value = choose(); edit(b => { if ('model' in b) b.model = $('pgModel').value; }); }
        loaded = true; modelStatus();
      } catch (e) { $('pgModelStatus').textContent = t('pg_models_failed') + ' ' + e.message; }
      finally { loading = false; $('pgRefresh').disabled = running; }
    }
    async function dataURL(file) {
      return new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = () => reject(new Error('pg_file_failed')); reader.readAsDataURL(file); });
    }
    async function send(event) {
      event.preventDefault(); if (running) return;
      // Snapshot everything before an asynchronous file read or model-list refresh.
      const e = selected;
      let b, headers, path = e.path;
      try {
        if (inputError) throw new Error(inputError);
        b = body(); if (e.method === 'POST' && !b) throw new Error('pg_invalid_json');
        headers = extraHeaders($('pgHeaders').value, $('pgKey').value);
        if (e.id === 'release') {
          if (!$('pgConfirm').checked || !/^[A-Za-z0-9_.-]{1,128}$/.test($('pgHandle').value)) throw new Error('pg_confirm_required');
          path += encodeURIComponent($('pgHandle').value);
        }
        if (b?.model && typeof b.model !== 'string') throw new Error('pg_invalid_json');
        if (e.id === 'speech' && b?.model === 'qwen3-tts-0.6b-customvoice' && b.instructions) throw new Error('pg_style_17');
      } catch (err) { const code = err.message.startsWith('pg_') ? err.message : 'pg_invalid_json'; clear(); resultState = 'failed'; editError = code; renderStatus(); return; }
      const files = [...$('pgFiles').files];
      clear(); busy(true); controller = new AbortController(); resultState = 'preparing'; renderStatus();
      try {
        const requestBody = await preparePayload(e, b, files, dataURL);
        if (controller.signal.aborted) throw new DOMException('Aborted', 'AbortError');
        if (b && !e.multipart) headers['Content-Type'] = 'application/json';
        started = performance.now(); resultState = 'running'; renderStatus();
        timer = setInterval(() => { $('pgTotal').textContent = seconds(performance.now() - started); }, 100);
        const response = await apiFetch(path, {method: e.method, headers, body: requestBody, signal: controller.signal, cache: 'no-store', redirect: 'error'});
        statusCode = String(response.status) + ' ' + response.statusText;
        $('pgResponseHeaders').textContent = [...response.headers].map(([k, v]) => k + ': ' + v).join('\n');
        const type = response.headers.get('Content-Type') || '', audio = response.ok && (type.startsWith('audio/') || (e.id === 'speech' && type.includes('octet-stream')));
        const stream = type.includes('text/event-stream'), chunks = [], textChunks = [], decoder = new TextDecoder();
        let first = true, textLength = 0;
        const parser = sseParser(value => { const part = answer(value); if (part) { output = (output + part).slice(0, MAX_TEXT); $('pgAnswer').textContent = output; $('pgAnswer').hidden = false; } });
        const reader = response.body.getReader();
        while (true) {
          const {value, done} = await reader.read(); if (done) break;
          if (first && value.length) { $('pgFirstByte').textContent = seconds(performance.now() - started); first = false; }
          received += value.length; $('pgReceived').textContent = bytes(received);
          if (received > MAX_RESPONSE) { await reader.cancel(); throw new Error('pg_response_limit'); }
          if (audio) chunks.push(value);
          else {
            const text = decoder.decode(value, {stream: true});
            textLength += text.length;
            if (!stream) textChunks.push(text);
            if (raw.length < MAX_TEXT) raw = (raw + text).slice(0, MAX_TEXT);
            if (stream) parser.push(text);
          }
        }
        if (!audio) { const tail = decoder.decode(); raw = (raw + tail).slice(0, MAX_TEXT); if (stream) { parser.push(tail, true); $('pgRaw').textContent = raw; } else textChunks.push(tail); }
        if (audio) {
          let blob = new Blob(chunks, {type});
          const pcm = b?.response_format === 'pcm';
          if (pcm) { download(blob, 'speech.pcm'); blob = new Blob([pcmWav(new Uint8Array(await blob.arrayBuffer()))], {type: 'audio/wav'}); }
          download(blob, 'speech.wav', 'audio');
          $('pgRaw').textContent = t('pg_audio_result', {bytes: received});
        } else if (!stream) {
          let value; try { value = JSON.parse(textChunks.join('')); } catch { value = null; }
          if (value && response.ok) imageResults(value);
          output = value ? answer(value) : raw;
          $('pgAnswer').textContent = (output || (value ? JSON.stringify(value, (k, v) => k === 'b64_json' ? '[image data]' : v, 2) : raw)).slice(0, MAX_TEXT);
          $('pgAnswer').hidden = false;
          $('pgRaw').textContent = value ? JSON.stringify(value, (k, v) => k === 'b64_json' ? '[image data]' : v, 2).slice(0, MAX_TEXT) : raw;
        }
        if (textLength > MAX_TEXT) { $('pgRaw').textContent += '\n' + t('pg_truncated'); }
        resultState = response.ok ? 'complete' : 'failed';
      } catch (error) {
        resultState = controller.signal.aborted || error.name === 'AbortError' ? 'cancelled' : 'failed';
        if (resultState === 'failed') { editError = error.message.startsWith('pg_') ? error.message : 'pg_network_error'; $('pgRaw').textContent += '\n' + error.message; }
      } finally {
        clearInterval(timer);
        if (started) $('pgTotal').textContent = seconds(performance.now() - started);
        if (raw && !$('pgRaw').textContent) $('pgRaw').textContent = raw;
        controller = null; busy(false); renderStatus();
      }
    }
    $('pgEndpoint').addEventListener('change', reset); $('pgReset').addEventListener('click', reset);
    $('pgRefresh').addEventListener('click', refreshModels); $('pgClear').addEventListener('click', clear);
    $('pgForm').addEventListener('submit', send); $('pgStop').addEventListener('click', () => controller?.abort());
    $('pgInput').addEventListener('input', () => edit(b => setText(selected, b, $('pgInput').value)));
    $('pgModel').addEventListener('input', () => edit(b => { b.model = $('pgModel').value; }));
    $('pgStream').addEventListener('change', () => edit(b => { b.stream = $('pgStream').checked; if (selected.id === 'speech') b.response_format = b.stream ? 'pcm' : 'wav'; }));
    $('pgSize').addEventListener('change', () => edit(b => { b.size = $('pgSize').value; }));
    for (const [id, key] of [['pgVoice', 'voice'], ['pgLanguage', 'language'], ['pgInstructions', 'instructions']]) $(id).addEventListener('input', () => edit(b => { if ($(id).value) b[key] = $(id).value; else delete b[key]; }));
    $('pgBody').addEventListener('input', () => {
      try { const b = body(); inputError = ''; if (b) { $('pgInput').value = textValue(selected, b); if (typeof b.model === 'string') $('pgModel').value = b.model; $('pgStream').checked = !!b.stream; if (b.size) $('pgSize').value = b.size; if (b.voice) $('pgVoice').value = b.voice; if (b.language) $('pgLanguage').value = b.language; $('pgInstructions').value = b.instructions || ''; } }
      catch { inputError = 'pg_invalid_json'; }
    });
    window.addEventListener('pagehide', () => { controller?.abort(); for (const url of urls) URL.revokeObjectURL(url); });
    reset();
    return {activate() { if (!loaded && !loading) refreshModels(); }, translate};
  }
  const api = {ENDPOINTS, STREAM_IDS, sample, textValue, setText, pcmWav, sseParser, answer, extraHeaders, preparePayload, create};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.HalobridgePlayground = api;
})(typeof window !== 'undefined' ? window : globalThis);
