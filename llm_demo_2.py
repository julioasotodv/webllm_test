import asyncio
import io
import panel as pn
import param

from panel.custom import JSComponent, ESMEvent

pn.extension('mathjax', template='material')


class WebLLM(JSComponent):

    loaded = param.Boolean(default=False, doc="""
        Whether the model is loaded.""")

    history = param.Integer(default=3)

    status = param.Dict(default={'text': '', 'progress': 0})

    running = param.Boolean(default=False, doc="""
        Whether the LLM is currently running.""")
    
    temperature = param.Number(default=1, bounds=(0, 2), doc="""
        Temperature of the model completions.""")

    _esm = """
    import * as webllm from "https://esm.run/@mlc-ai/web-llm";

    export async function render({ model }) {

        const initProgressCallback = (status) => {
              model.status = status
            }

        const engine = await webllm.CreateMLCEngine(
                    "gemma-2-9b-it-q4f16_1-MLC",
                   {initProgressCallback}
                )

        model.loaded = true;
        model.on("msg:custom", async (event) => {

    if (event.type === 'completion') {

    const chunks = await engine.chat.completions.create({
        messages: event.messages,
        temperature: 1,
        stream: true, // <-- Enable streaming
        //stream_options: { include_usage: true },
    });

    model.running = true

    for await (const chunk of chunks) {
        if (!model.running) {
              break
            }
         model.send_msg(chunk.choices[0])
         }
        }
    })
    }
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.loading = True
        self._buffer = []

    @param.depends('loaded', watch=True)
    def _loaded(self):
        self.loading = False

    def _handle_msg(self, msg):
        if self.running:
            self._buffer.insert(0, msg)

    async def create_completion(self, msgs):
        self._send_msg({'type': 'completion', 'messages': msgs})
        latest = None
        while True:
            await asyncio.sleep(0.01)
            if not self._buffer:
                continue
            choice = self._buffer.pop()
            yield choice
            reason = choice['finish_reason']
            if reason == 'error':
                raise RuntimeError('Model not loaded')
            elif reason:
                return

    async def callback(self, contents: str, user: str):
        if not self.loaded:
            if self.loading:
                yield pn.pane.Markdown(
                    f'## `model`\n\n' + self.param.status.rx()['text']
                )
            else:
                yield 'Load the model'
            return
        self.running = False
        self._buffer.clear()
        message = ""
        async for chunk in llm.create_completion([{'role': 'user', 'content': contents}]):
            message += chunk['delta'].get('content', '')
            yield message

    def menu(self):
        status = self.param.status.rx()
        return pn.Column(
            pn.widgets.FloatSlider.from_param(self.param.temperature, sizing_mode='stretch_width'),
            pn.indicators.Progress(
                value=(status['progress']*100).rx.pipe(int), visible=self.param.loading,
                sizing_mode='stretch_width'
            ),
            pn.pane.Markdown(status['text'], visible=self.param.loading)
        )

llm = WebLLM()

pn.Column(
    llm.menu(),
    llm
).servable(area='sidebar')

boton_subir_pdf = pn.widgets.FileInput(accept=".pdf,.txt", disabled=True)

chat_feed = pn.chat.ChatFeed(callback=llm.callback)


def activar_boton_pdf(*events):
    for event in events:
        if event.name == "loaded" and event.new == True:
            boton_subir_pdf.disabled = False

llm.param.watch(activar_boton_pdf, 'loaded')

def get_model_answer(*events):
    for event in events:
        if event.name == "value":
            if event.new is not None:
                uploaded_pdf = io.BytesIO(event.new)
                texto = uploaded_pdf.read().decode("utf-8")
                prompt = pn.chat.ChatMessage(texto, visible=False)
                chat_feed.send(prompt, respond=True)

boton_subir_pdf.param.watch(get_model_answer, 'value')

pn.Column(boton_subir_pdf, pn.Row(chat_feed)).servable(title='WebLLM')
