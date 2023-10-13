import shap
import streamlit as st
import streamlit.components.v1 as components
from transformers import AutoTokenizer, AutoModelForCausalLM

@st.cache_resource
def load_tokenizer_and_model():
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    return tokenizer, model

tokenizer, model = load_tokenizer_and_model()

@st.cache_resource
def get_shap_values():
    prompt = """
    The 2008 Summer Olympics torch relay was run from March 24 until August 8, 2008, prior to the 2008 Summer Olympics, with the theme of "one world, one dream". Plans for the relay were announced on April 26, 2007, in Beijing, China. The relay, also called by the organizers as the "Journey of Harmony", lasted 129 days and carried the torch 137,000 km (85,000 mi) – the longest distance of any Olympic torch relay since the tradition was started ahead of the 1936 Summer Olympics.

    Q: What was the theme
    A:
    """.strip()
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    # Sample instead of returning the most likely token
    model.config.do_sample=False
    # Set maximum length
    model.config.max_new_tokens = 10
    # Generate text with stop_token set to "."
    output = model.generate(input_ids)
    st.write("Hello 1")
    output_text = tokenizer.decode(output[0][-10:], skip_special_tokens=True)
    
    model.config.is_decoder = True
    explainer = shap.Explainer(model, tokenizer)
    shap_values = explainer([prompt])
    st.write("Hello 2")
    return shap_values

shap_values = get_shap_values()

from streamlit_shap import st_shap

st_shap(shap.plots.text(shap_values), height=500)
