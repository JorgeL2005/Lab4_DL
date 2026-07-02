from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TextStreamer, StoppingCriteria, StoppingCriteriaList)
import torch

MODEL_ID = "Qwen/Qwen3-8B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.eval()


class StopOnText(StoppingCriteria):
    """Detiene la generacion en cuanto aparece `stop` en el texto recien generado."""
    def __init__(self, tokenizer, prompt_len, stop="[PLAN END]"):
        self.tok = tokenizer
        self.prompt_len = prompt_len
        self.stop = stop

    def __call__(self, input_ids, scores, **kw):
        text = self.tok.decode(input_ids[0, self.prompt_len:], skip_special_tokens=True)
        return self.stop in text


def qwen( prompt:str,
          system:str|None=None,
          max_new_tokens:int=512,
          temperature:float=0.8,
          top_p:float=0.9,
          enable_thinking:bool=False,
          do_sample:bool=False,
          stream:bool=False) -> str:

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # ID de tokens (respuesta)
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking,
    )
    # Traducimos los Ids como texto
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    plen = inputs.input_ids.shape[1]
    stops = StoppingCriteriaList([StopOnText(tokenizer, plen)])

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=stops,
    )
    # En modo determinista (greedy) no pasamos temperature/top_p para evitar warnings.
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    if stream:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        _ = model.generate(**inputs, streamer=streamer, **gen_kwargs)
        return ""

    with torch.no_grad():
        # resp = [input, output]
        out = model.generate(**inputs, **gen_kwargs)
    # Sólo la parte nueva:
    gen_ids = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)