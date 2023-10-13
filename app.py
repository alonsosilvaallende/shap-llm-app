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
    
    After being lit at the birthplace of the Olympic Games in Olympia, Greece on March 24, the torch trav- eled to the Panathinaiko Stadium in Athens, and then to Beijing, arriving on March 31. From Beijing, the torch was following a route passing through six continents. The torch has visited cities along the Silk Road, symbolizing ancient links between China and the rest of the world. The relay also included an ascent with the flame to the top of Mount Everest on the border of Nepal and Tibet, China from the Chinese side, which was closed specially for the event.
    
    Q: What was the theme
    A:
    """.strip()
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    # Sample instead of returning the most likely token
    model.config.do_sample=False
    # Set maximum length
    model.config.max_new_tokens = 50
    # Generate text with stop_token set to "."
    output = model.generate(input_ids)
    
    output_text = tokenizer.decode(output[0][-50:], skip_special_tokens=True)
    
    model.config.is_decoder = True
    explainer = shap.Explainer(model, tokenizer)
    shap_values = explainer([prompt])
    return shap_values

shap_values = get_shap_values()

from streamlit_shap import st_shap

st_shap(shap.plots.text(shap_values), height=500)
