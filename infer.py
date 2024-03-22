from transformers import AutoModel
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch


model = AutoModel.from_pretrained("models/chatglm-6b", trust_remote_code=True, load_in_8bit=True, device_map='auto')

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("models/chatglm-6b", trust_remote_code=True)

from peft import PeftModel

model = PeftModel.from_pretrained(model, "./output")

import json

instructions = json.load(open("data/alpaca_data.json"))

answers = []
from cover_alpaca2jsonl import format_example


with torch.no_grad():
    for idx, item in enumerate(instructions[:3]):
        feature = format_example(item)
        input_text = feature['context']
        ids = tokenizer.encode(input_text)
        input_ids = torch.LongTensor([ids])
        out = model.generate(
            input_ids=input_ids,
            max_length=150,
            do_sample=False,
            temperature=0
        )
        out_text = tokenizer.decode(out[0])
        answer = out_text.replace(input_text, "").replace("\nEND", "").strip()
        item['infer_answer'] = answer
        print(out_text)
        print(f"### {idx+1}.Answer:\n", item.get('output'), '\n\n')
        answers.append({'index': idx, **item})