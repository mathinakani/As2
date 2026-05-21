
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import textwrap

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="International Business Marketing AI",
    page_icon="🌍",
    layout="wide"
)

# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.title {
    font-size:40px;
    font-weight:bold;
    color:#0f766e;
    text-align:center;
}
.subtitle {
    font-size:18px;
    text-align:center;
    color:#444;
}
.result-box {
    background-color:white;
    padding:20px;
    border-radius:12px;
    box-shadow:0px 0px 10px rgba(0,0,0,0.1);
    margin-top:15px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# HEADER
# ------------------------------------------------
st.markdown('<div class="title">🌍 International Business Marketing Prompt Application</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Generate Global Marketing Titles, Slogans, and Expert Advertising Content using Hugging Face AI</div>',
    unsafe_allow_html=True
)

st.write("")

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512
    )
    return generator

generator = load_model()

# ------------------------------------------------
# USER INPUT
# ------------------------------------------------
product_name = st.text_input(
    "Enter Product Name",
    placeholder="Example: Smart Water Bottle"
)

target_market = st.selectbox(
    "Select Target Global Market",
    [
        "Global",
        "USA",
        "Europe",
        "Asia",
        "Middle East",
        "India",
        "Africa"
    ]
)

marketing_style = st.selectbox(
    "Select Marketing Style",
    [
        "Professional",
        "Luxury",
        "Emotional",
        "Corporate",
        "Modern",
        "Youth",
        "Premium"
    ]
)

# ------------------------------------------------
# GENERATE BUTTON
# ------------------------------------------------
if st.button("🚀 Generate Marketing Content"):

    if product_name.strip() == "":
        st.warning("Please enter a product name.")
    else:

        with st.spinner("Generating AI Marketing Content..."):

            # ----------------------------------------
            # MASTER PROMPT
            # ----------------------------------------
            prompt = f"""
You are an international business marketing expert.

Create professional international advertising content for the following product.

Product Name: {product_name}

Target Market: {target_market}

Marketing Style: {marketing_style}

Generate the following:

1. Global-Ready Product Title
2. Powerful International Marketing Slogan
3. Advertising Description from THREE perspectives:
   - Emotional Marketing Expert
   - Corporate Branding Expert
   - Digital Advertising Expert

Requirements:
- Professional business language
- International audience compatibility
- Persuasive branding techniques
- Emotional engagement
- Premium marketing quality
- Easy to understand globally
- Attractive and modern tone

Format clearly with headings.
"""

            # ----------------------------------------
            # GENERATE RESPONSE
            # ----------------------------------------
            result = generator(prompt)

            output = result[0]["generated_text"]

        # ----------------------------------------
        # DISPLAY RESULTS
        # ----------------------------------------
        st.markdown("## ✨ Generated Marketing Content")

        st.markdown(
            f"""
            <div class="result-box">
            <pre style="white-space: pre-wrap; font-size:16px;">{output}</pre>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------
        # DOWNLOAD OPTION
        # ----------------------------------------
        st.download_button(
            label="📥 Download Result",
            data=output,
            file_name="marketing_content.txt",
            mime="text/plain"
        )

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
st.sidebar.title("📌 About Project")

st.sidebar.info("""
This AI application uses Hugging Face Large Language Models
to generate:

✅ Global Product Titles

✅ Marketing Slogans

✅ Expert Advertising Descriptions

Technologies Used:
- Streamlit
- Hugging Face Transformers
- FLAN-T5 Large
- Prompt Engineering
""")

st.sidebar.success("Developed for International Business Marketing Assignment")
