const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const p = require('../src/halobridge_data/playground.js');
const endpoint = id => p.ENDPOINTS.find(e => e.id === id);

test('Liquid multimodal requests retain audio, frame order, and spoken response bytes', async () => {
  const read = async f => f.name.endsWith('.wav') ? 'data:audio/wav;base64,UklGRg==' : 'data:image/png;base64,' + f.name;
  const wav = {name:'talk.wav',size:4};
  const e = endpoint('npu_lfm3_voice');
  const body = JSON.parse(await p.preparePayload(e,p.sample(e,'own-omni'),[wav],read));
  assert.deepEqual(body.modalities,['text','audio']);
  assert.equal(body.messages[0].content[1].input_audio.data,'UklGRg==');
  const video = endpoint('npu_lfm3_video'), b = p.sample(video,'own-vl');
  b.messages[0].content[1].fps = 24;
  p.setText(video,b,'New prompt');
  const encoded = JSON.parse(await p.preparePayload(video,b,[{name:'1.png',size:1},{name:'2.png',size:1}],read));
  assert.equal(encoded.messages[0].content[1].fps,24);
  assert.equal(encoded.messages[0].content[0].text,'New prompt');
  assert.deepEqual(encoded.messages[0].content[1].frames,['data:image/png;base64,1.png','data:image/png;base64,2.png']);
  const reply = {choices:[{message:{audio:{data:'AQIDBA==',transcript:'Hello'}}}]};
  assert.deepEqual(p.audioChunks(reply),[new Uint8Array([1,2,3,4])]);
  assert.equal(p.answer(reply),'Hello');
  const delta = {choices:[{delta:{audio:{data:'BQg=',transcript:' world'}}}]};
  assert.deepEqual(p.audioChunks(delta),[new Uint8Array([5,8])]);
  assert.equal(p.answer(delta),' world');
});

test('presets cover all documented inference routes and stay on this router', () => {
  const routes = new Set(p.ENDPOINTS.map(e => e.method + ' ' + e.path));
  for (const route of ['POST /v1/chat/completions', 'POST /v1/completions', 'POST /v1/responses',
    'POST /v1/messages', 'POST /v1/messages/count_tokens', 'POST /v1/systemone', 'POST /v1/embeddings',
    'POST /v1/rerank', 'POST /v1/moderations', 'POST /classify', 'POST /v1/images/generations',
    'POST /v1/images/edits', 'POST /v1/images/variations', 'POST /v1/audio/transcriptions', 'POST /v1/audio/speech',
    'GET /v1/models', 'GET /health', 'GET /metrics', 'GET /cache', 'GET /props', 'GET /slots',
    'POST /tokenize', 'POST /detokenize', 'GET /halogen/handles', 'DELETE /halogen/handles/']) assert.ok(routes.has(route), route);
  for (const e of p.ENDPOINTS) assert.match(e.path, /^\/[a-z0-9/_]+$/);
  const ids = p.ENDPOINTS.map(e => e.id); assert.equal(new Set(ids).size, ids.length);
  for (const lang of ['en', 'de']) {
    const strings = JSON.parse(fs.readFileSync(path.join(__dirname, '../src/halobridge_data/locales', lang + '.json'), 'utf8'));
    for (const e of p.ENDPOINTS) assert.equal(typeof strings['pg_example_' + e.id], 'string');
  }
});

test('editing the primary prompt retains tools, schemas and model settings', () => {
  for (const id of ['chat', 'tools', 'structured', 'decision', 'responses', 'messages', 'completions', 'systemone', 'rerank', 'embedding', 'moderations', 'speech']) {
    const e = endpoint(id), body = p.sample(e, 'model-id');
    p.setText(e, body, 'A different input');
    assert.equal(p.textValue(e, body), 'A different input', id);
    assert.equal(body.model, 'model-id');
    if (id === 'tools') assert.equal(body.tools[0].function.strict, true);
    if (id === 'structured') assert.equal(body.response_format.json_schema.strict, true);
  }
  assert.throws(() => p.setText(endpoint('detokenize'), {tokens: []}, '[1,-2]'), /pg_invalid_tokens/);
});

test('arbitrary SSE chunk boundaries preserve text, tools and unicode', () => {
  const events = [];
  const stream = 'event: message\r\ndata: {"choices":[{"delta":{"content":"Grüße"}}]}\r\n\r\n'
    + 'data: {"type":"response.output_text.delta","delta":"!"}\n\n'
    + 'data: {"type":"content_block_delta","delta":{"partial_json":"{\\"city\\":\\"Berlin\\"}"}}\n\n'
    + 'data: [DONE]\n\n';
  const parser = p.sseParser(e => events.push(p.answer(e)));
  for (const char of stream) parser.push(char);
  parser.push('', true);
  assert.deepEqual(events, ['Grüße', '!', '{"city":"Berlin"}']);
  assert.equal(p.answer({error: {message: 'NPU restarting'}}), 'NPU restarting');
});

test('PCM playback wraps exact 24kHz mono samples without modifying audio', () => {
  const pcm = new Uint8Array([0, 128, 255, 127]);
  const wav = p.pcmWav(pcm), view = new DataView(wav.buffer);
  assert.equal(Buffer.from(wav.slice(0, 4)).toString(), 'RIFF');
  assert.equal(view.getUint32(24, true), 24000);
  assert.equal(view.getUint16(22, true), 1);
  assert.equal(view.getUint16(34, true), 16);
  assert.equal(view.getUint32(40, true), pcm.length);
  assert.deepEqual(wav.slice(44), pcm);
  assert.throws(() => p.pcmWav(new Uint8Array([1])), /pg_invalid_pcm/);
});

test('four multipart images keep their bytes and upload order', async () => {
  const files = [1, 2, 3, 4].map(n => Object.assign(new Blob([new Uint8Array([137, 80, 78, 71, n])], {type: 'image/png'}), {name: `${n}.png`}));
  const form = await p.preparePayload(endpoint('edits'), {model: 'flux2-klein-4b', prompt: 'Combine images 1 to 4'}, files);
  assert.equal(form.get('model'), 'flux2-klein-4b');
  const uploads = form.getAll('image'); assert.equal(uploads.length, 4);
  for (let i = 0; i < files.length; i++) assert.deepEqual(new Uint8Array(await uploads[i].arrayBuffer()), new Uint8Array([137, 80, 78, 71, i + 1]));
  await assert.rejects(p.preparePayload(endpoint('edits'), {}, [...files, files[0]]), /pg_file_count/);
  await assert.rejects(p.preparePayload(endpoint('edits'), {}, []), /pg_file_count/);
});

test('embedding uploads use upstream multimodal input items', async () => {
  const read = async () => 'data:audio/wav;base64,UklGRg==';
  const wav = {name: 'speech.wav', size: 16};
  const audio = JSON.parse(await p.preparePayload(endpoint('embedding_audio'), {model: 'embeddinggemma-2'}, [wav], read));
  assert.deepEqual(audio.input, [{type: 'input_audio', input_audio: {data: 'UklGRg==', format: 'wav'}}]);
  const picture = JSON.parse(await p.preparePayload(endpoint('embedding_image'), {model: 'embeddinggemma-2'}, [{name: 'a.png', size: 16}], async () => 'data:image/png;base64,YQ=='));
  assert.deepEqual(picture.input, [{type: 'image_url', image_url: {url: 'data:image/png;base64,YQ=='}}]);
  await assert.rejects(p.preparePayload(endpoint('transcriptions'), {}, [{name: 'a.mp3', size: 1}]), /pg_wav_hint/);
  await assert.rejects(p.preparePayload(endpoint('vision'), {}, [{name: 'a.png', size: 26 * 1024 ** 2}]), /pg_file_size/);
});

test('vision uploads preserve the prompt and other chat parameters', async () => {
  const e = endpoint('vision'), b = p.sample(e, 'gpu-model');
  const result = JSON.parse(await p.preparePayload(e, b, [{name: 'a.png', size: 5}], async () => 'data:image/png;base64,YQ=='));
  assert.equal(result.messages[0].content[0].text, 'Describe this image.');
  assert.equal(result.messages[0].content[1].image_url.url, 'data:image/png;base64,YQ==');
  assert.equal(result.max_tokens, 512);
});

test('structured vision keeps the schema and d1 media retains frame order and fps', async () => {
  const picture = {name:'frame.png', size:5};
  const read = async file => 'data:image/png;base64,' + file.name;
  const vision = endpoint('vision_structured');
  const result = JSON.parse(await p.preparePayload(vision, p.sample(vision,'gpu'),[picture],read));
  assert.equal(result.response_format.json_schema.strict,true);
  assert.equal(result.messages[0].content[1].image_url.url,'data:image/png;base64,frame.png');
  const d1 = endpoint('systemone_d1_image');
  const boxed = JSON.parse(await p.preparePayload(d1,p.sample(d1,'d1-2B'),[picture],read));
  assert.equal(boxed.questions.objects.type,'boxes');
  assert.deepEqual(boxed.media,['data:image/png;base64,frame.png']);
  const video = endpoint('systemone_d1_video');
  const request = p.sample(video,'d1-2B'); request.media[0].fps=24;
  const frames = [picture,{name:'second.png',size:4}];
  const encoded = JSON.parse(await p.preparePayload(video,request,frames,read));
  assert.deepEqual(encoded.media,[{type:'video',frames:['data:image/png;base64,frame.png','data:image/png;base64,second.png'],fps:24}]);
  await assert.rejects(p.preparePayload(video,request,Array(5).fill(picture),read),/pg_file_count/);
  assert.ok(p.sample(endpoint('npu_lfm3'),'LFM3').tools.length);
  assert.ok(p.sample(endpoint('npu_lfm3_base'),'LFM3-Base').prompt);
});

test('custom headers cannot override destinations, cookies or credentials', () => {
  const h = p.extraHeaders('{"X-Halogen-Priority":"1","anthropic-version":"2023-06-01"}', 'private-test-key');
  assert.equal(h.Authorization, 'Bearer private-test-key');
  assert.equal(h['X-Halogen-Priority'], '1');
  for (const s of ['[]', 'null', '{"Host":"evil.invalid"}', '{"Cookie":"secret"}', '{"Authorization":"secret"}', '{"X-Halogen-Priority":1}', '{"X-Halogen-Keep":"a\\r\\nb"}']) assert.throws(() => p.extraHeaders(s, ''), /pg_invalid_headers/);
});
