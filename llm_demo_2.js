importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.0/full/pyodide.js");

function sendPatch(patch, buffers, msg_id) {
  self.postMessage({
    type: 'patch',
    patch: patch,
    buffers: buffers
  })
}

async function startApplication() {
  console.log("Loading pyodide!");
  self.postMessage({type: 'status', msg: 'Loading pyodide'})
  self.pyodide = await loadPyodide();
  self.pyodide.globals.set("sendPatch", sendPatch);
  console.log("Loaded!");
  await self.pyodide.loadPackage("micropip");
  const env_spec = ['https://cdn.holoviz.org/panel/wheels/bokeh-3.6.2-py3-none-any.whl', 'https://cdn.holoviz.org/panel/1.6.0/dist/wheels/panel-1.6.0-py3-none-any.whl', 'pyodide-http==0.2.1', 'param']
  for (const pkg of env_spec) {
    let pkg_name;
    if (pkg.endsWith('.whl')) {
      pkg_name = pkg.split('/').slice(-1)[0].split('-')[0]
    } else {
      pkg_name = pkg
    }
    self.postMessage({type: 'status', msg: `Installing ${pkg_name}`})
    try {
      await self.pyodide.runPythonAsync(`
        import micropip
        await micropip.install('${pkg}');
      `);
    } catch(e) {
      console.log(e)
      self.postMessage({
	type: 'status',
	msg: `Error while installing ${pkg_name}`
      });
    }
  }
  console.log("Packages loaded!");
  self.postMessage({type: 'status', msg: 'Executing code'})
  const code = `
  \nimport asyncio\n\nfrom panel.io.pyodide import init_doc, write_doc\n\ninit_doc()\n\nimport asyncio\nimport io\nimport panel as pn\nimport param\n\nfrom panel.custom import JSComponent, ESMEvent\n\npn.extension('mathjax', template='material')\n\n\nclass WebLLM(JSComponent):\n\n    loaded = param.Boolean(default=False, doc="""\n        Whether the model is loaded.""")\n\n    history = param.Integer(default=3)\n\n    status = param.Dict(default={'text': '', 'progress': 0})\n\n    running = param.Boolean(default=False, doc="""\n        Whether the LLM is currently running.""")\n    \n    temperature = param.Number(default=1, bounds=(0, 2), doc="""\n        Temperature of the model completions.""")\n\n    _esm = """\n    import * as webllm from "https://esm.run/@mlc-ai/web-llm";\n\n    export async function render({ model }) {\n\n        const initProgressCallback = (status) => {\n              model.status = status\n            }\n\n        const engine = await webllm.CreateMLCEngine(\n                    "gemma-2-9b-it-q4f16_1-MLC",\n                   {initProgressCallback}\n                )\n\n        model.loaded = true;\n        model.on("msg:custom", async (event) => {\n\n    if (event.type === 'completion') {\n\n    const chunks = await engine.chat.completions.create({\n        messages: event.messages,\n        temperature: 1,\n        stream: true, // <-- Enable streaming\n        //stream_options: { include_usage: true },\n    });\n\n    model.running = true\n\n    for await (const chunk of chunks) {\n        if (!model.running) {\n              break\n            }\n         model.send_msg(chunk.choices[0])\n         }\n        }\n    })\n    }\n    """\n\n    def __init__(self, **params):\n        super().__init__(**params)\n        self.loading = True\n        self._buffer = []\n\n    @param.depends('loaded', watch=True)\n    def _loaded(self):\n        self.loading = False\n\n    def _handle_msg(self, msg):\n        if self.running:\n            self._buffer.insert(0, msg)\n\n    async def create_completion(self, msgs):\n        self._send_msg({'type': 'completion', 'messages': msgs})\n        latest = None\n        while True:\n            await asyncio.sleep(0.01)\n            if not self._buffer:\n                continue\n            choice = self._buffer.pop()\n            yield choice\n            reason = choice['finish_reason']\n            if reason == 'error':\n                raise RuntimeError('Model not loaded')\n            elif reason:\n                return\n\n    async def callback(self, contents: str, user: str):\n        if not self.loaded:\n            if self.loading:\n                yield pn.pane.Markdown(\n                    f'## \`model\`\\n\\n' + self.param.status.rx()['text']\n                )\n            else:\n                yield 'Load the model'\n            return\n        self.running = False\n        self._buffer.clear()\n        message = ""\n        async for chunk in llm.create_completion([{'role': 'user', 'content': contents}]):\n            message += chunk['delta'].get('content', '')\n            yield message\n\n    def menu(self):\n        status = self.param.status.rx()\n        return pn.Column(\n            pn.widgets.FloatSlider.from_param(self.param.temperature, sizing_mode='stretch_width'),\n            pn.indicators.Progress(\n                value=(status['progress']*100).rx.pipe(int), visible=self.param.loading,\n                sizing_mode='stretch_width'\n            ),\n            pn.pane.Markdown(status['text'], visible=self.param.loading)\n        )\n\nllm = WebLLM()\n\npn.Column(\n    llm.menu(),\n    llm\n).servable(area='sidebar')\n\nboton_subir_pdf = pn.widgets.FileInput(accept=".pdf,.txt", disabled=True)\n\nchat_feed = pn.chat.ChatFeed(callback=llm.callback)\n\n\ndef activar_boton_pdf(*events):\n    for event in events:\n        if event.name == "loaded" and event.new == True:\n            boton_subir_pdf.disabled = False\n\nllm.param.watch(activar_boton_pdf, 'loaded')\n\ndef get_model_answer(*events):\n    for event in events:\n        if event.name == "value":\n            if event.new is not None:\n                uploaded_pdf = io.BytesIO(event.new)\n                texto = uploaded_pdf.read().decode("utf-8")\n                prompt = pn.chat.ChatMessage(texto, visible=False)\n                chat_feed.send(prompt, respond=True)\n\nboton_subir_pdf.param.watch(get_model_answer, 'value')\n\npn.Column(boton_subir_pdf, pn.Row(chat_feed)).servable(title='WebLLM')\n\n\nawait write_doc()
  `

  try {
    const [docs_json, render_items, root_ids] = await self.pyodide.runPythonAsync(code)
    self.postMessage({
      type: 'render',
      docs_json: docs_json,
      render_items: render_items,
      root_ids: root_ids
    })
  } catch(e) {
    const traceback = `${e}`
    const tblines = traceback.split('\n')
    self.postMessage({
      type: 'status',
      msg: tblines[tblines.length-2]
    });
    throw e
  }
}

self.onmessage = async (event) => {
  const msg = event.data
  if (msg.type === 'rendered') {
    self.pyodide.runPythonAsync(`
    from panel.io.state import state
    from panel.io.pyodide import _link_docs_worker

    _link_docs_worker(state.curdoc, sendPatch, setter='js')
    `)
  } else if (msg.type === 'patch') {
    self.pyodide.globals.set('patch', msg.patch)
    self.pyodide.runPythonAsync(`
    from panel.io.pyodide import _convert_json_patch
    state.curdoc.apply_json_patch(_convert_json_patch(patch), setter='js')
    `)
    self.postMessage({type: 'idle'})
  } else if (msg.type === 'location') {
    self.pyodide.globals.set('location', msg.location)
    self.pyodide.runPythonAsync(`
    import json
    from panel.io.state import state
    from panel.util import edit_readonly
    if state.location:
        loc_data = json.loads(location)
        with edit_readonly(state.location):
            state.location.param.update({
                k: v for k, v in loc_data.items() if k in state.location.param
            })
    `)
  }
}

startApplication()