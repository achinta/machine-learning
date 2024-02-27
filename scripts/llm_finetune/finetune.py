from huggingface_hub import hf_hub_download
from datasets import load_dataset, Dataset
from typer import Typer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

model_name = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
model_file = "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
dataset_id = "b-mc2/sql-create-context"

model_path = hf_hub_download(model_name, filename=model_file)
print(f"Model downloaded to {model_path}")
# print the file size using pathlib
print(f"File size: {Path(model_path).stat().st_size / (1024*1024):.2f} MB")

data = load_dataset(dataset_id, split='train')
df = data.to_pandas()

def get_prompt(context, question, answer='', add_gen_prompt=True):
    gen_prompt = "<|answer|>: {answer}"
    template = '''
<|context|>: {context}
<|question|>: {question}'''
    if add_gen_prompt:
        template += gen_prompt
    return template.format(context=context, question=question, answer=answer)

df['text'] = df.apply(lambda x: get_prompt(x['context'], x['question'], x['answer'], add_gen_prompt=True), axis=1)
dataset = Dataset.from_pandas(df)

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# training
from peft import LoraConfig, PeftConfig
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
trainer = SFTTrainer(model_id, train_dataset=dataset, tokenizer=tokenizer, dataset_text_field="text", packing=False, 
                     peft_config=peft_config)
trainer.train()