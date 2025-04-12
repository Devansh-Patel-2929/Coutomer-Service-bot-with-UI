from transformers import T5Tokenizer, T5ForConditionalGeneration
import gradio as gr
import re

tokenizer = T5Tokenizer.from_pretrained('chatbot_model')
model = T5ForConditionalGeneration.from_pretrained('chatbot_model')

def clean_input(text):
    return re.sub(r'@\w+|http\S+', '', text).strip()

def respond(message, history):
    message = clean_input(message)
    inputs = tokenizer.encode(
        message,
        return_tensors='pt',
        max_length=128,
        truncation=True
    )
    outputs = model.generate(
        inputs,
        max_length=160,
        num_beams=5,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

gr.ChatInterface(respond).launch()