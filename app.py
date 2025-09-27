import streamlit as st
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import json
import ast

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="เพื่อนใจวัยเรียน AI",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 เพื่อนใจวัยเรียน AI")
st.caption("AI Chatbot ที่พร้อมรับฟังและให้คำปรึกษาปัญหาของนักเรียน/นักศึกษา")

# --- 2. Model Loading Function ---
@st.cache_resource
def load_llm_model():
    """
    Downloads and loads the GGUF model from Hugging Face Hub.
    The model is cached to avoid reloading on every interaction.
    """
    try:
        print("--- DEBUG: Attempting to load model... ---")
        hf_token = st.secrets["HF_TOKEN"]
        
        print("--- DEBUG: Starting model download from Hugging Face Hub... ---")
        model_path = hf_hub_download(
            repo_id="openthaigpt/openthaigpt1.5-7b-instruct", # <--- **REPLACE WITH YOUR REPO ID**
            filename="openthaigpt1.5-7B-instruct-Q4KM.gguf", # <--- **REPLACE WITH YOUR GGUF FILENAME**
            token=hf_token
        )
        print(f"--- DEBUG: Model downloaded to: {model_path} ---")

        print("--- DEBUG: Loading model into memory with Llama... ---")
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=4096,
            n_batch=1024,
            verbose=False
        )
        print("--- DEBUG: Model loaded successfully! ---")
        return llm
    except Exception as e:
        print(f"--- DEBUG: CRITICAL ERROR during model loading: {e} ---")
        st.error(f"Error loading model: {e}")
        return None

# --- 3. Chat Logic Functions (Refactored) ---

def get_initial_answer(prompt: str, model):
    """
    Generates the initial, empathetic response from the model as a stream.
    This is the primary "Student's Mind Mate AI" persona.
    """
    if not model:
        yield "ขออภัยครับ ระบบ AI ไม่พร้อมใช้งานในขณะนี้ โปรดลองอีกครั้งในภายหลัง"
        return

    system_prompt = (
        "คุณคือ 'เพื่อนใจวัยเรียน AI' มีความเข้าอกเข้าใจ, ให้กำลังใจ, และไม่ตัดสิน โดยมีหน้าที่หลักดังนี้:\n"
        "1. เป็นเพื่อนรับฟังปัญหาและความเครียดของนักเรียน/นักศึกษา\n"
        "2. ช่วยให้คำแนะนำในการวางแผนการเรียน จัดการเวลา หรือเทคนิคที่ช่วยให้เรียนได้ดีขึ้น\n"
        "3. **สำคัญที่สุด:** หากผู้ใช้แสดงความเสี่ยงเกี่ยวกับการทำร้ายร่างกายตัวเอง ให้หยุดการให้คำแนะนำทั่วไปและแนะนำให้ติดต่อผู้เชี่ยวชาญทันที"
    )
    
    full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    print(f"\n--- DEBUG (Initial Answer): Sending prompt to model ---\n{full_prompt}\n-----------------------------------------------------\n")

    stream = model(
        full_prompt,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        stop=["<|im_end|>"],
        stream=True
    )
    
    for output in stream:
        yield output['choices'][0]['text']

def run_fact_checking_layer(user_prompt: str, initial_answer: str, model) -> dict:
    """
    Runs a second pass on the generated answer to check for safety and factual correctness.
    Returns a dictionary with the verdict.
    """
    if not model:
        return {"is_safe_and_correct": False, "explanation": "Model not available for fact-checking."}

    fact_check_system_prompt = (
        "คุณคือผู้ตรวจสอบข้อเท็จจริง AI ที่เชี่ยวชาญด้านสุขภาพจิต ภารกิจของคุณคือตรวจสอบความถูกต้องและความปลอดภัยของคำตอบที่ AI สร้างขึ้น\n"
        "โปรดวิเคราะห์คำตอบตามเกณฑ์ต่อไปนี้ และให้ผลลัพธ์ในรูปแบบ JSON เท่านั้น:\n"
        "1. **ความถูกต้องของข้อมูล (Factual Correctness):** เนื้อหาในคำตอบต้องเป็นข้อมูลที่ถูกต้องและไม่สร้างความเข้าใจผิด\n"
        "2. **ความปลอดภัย (Safety):** คำตอบจะต้องไม่มีเนื้อหาที่เป็นอันตราย, ยั่วยุ, หรือแนะนำสิ่งที่อาจส่งผลให้ผู้ใช้รู้สึกแย่ลงหรือเสี่ยงที่จะทำร้ายร่างกายตัวเองโดยเด็ดขาด\n\n"
        "**โครงสร้างผลลัพธ์ (JSON Output):**\n"
        "{\n"
        "  \"is_factually_correct\": boolean (true or false) // ตั้งค่าเป็น false หากไม่ผ่านเกณฑ์ข้อ 1 หรือ 2\n"
        "}"
    )


    fact_check_user_prompt = (
        f"**คำถามของผู้ใช้:**\n{user_prompt}\n\n"
        f"**คำตอบของ AI ที่ต้องประเมิน:**\n{initial_answer}\n\n"
        "โปรดประเมินคำตอบของ AI ตามเกณฑ์ที่กำหนดและตอบกลับในรูปแบบ JSON เท่านั้น"
    )

    full_prompt = (
        f"<|im_start|>system\n{fact_check_system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{fact_check_user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    print(f"\n--- DEBUG (Fact Check): Sending prompt to model ---\n{full_prompt}\n----------------------------------------------------\n")

    output = model(
        full_prompt,
        max_tokens=256,
        temperature=0.0,
        stop=["<|im_end|>", "```"],
        stream=False
    )
    print(f"--- DEBUG (Fact Check): Raw output from model ---\n{output}\n--------------------------------------------------\n")
    
    response_text = output['choices'][0]['text'].strip()
    print(f"--- DEBUG (Fact Check): Raw response from model ---\n{response_text}\n--------------------------------------------------\n")
    
    # --- START of requested change ---
    # **IMPROVED PARSING LOGIC** - COMMENTED OUT
    try:
        # 1. Clean up the response to extract the core text
        if "```json" in response_text:
            dict_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            dict_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            dict_str = response_text
        
        print(f"--- DEBUG (Fact Check): Extracted dict string ---\n{dict_str}\n--------------------------------------------------\n")
        
        # 2. First try to parse as JSON (handles lowercase true/false)
        try:
            parsed_dict = json.loads(dict_str)
            print("--- DEBUG (Fact Check): Successfully parsed as JSON ---")
        except json.JSONDecodeError:
            print("--- DEBUG (Fact Check): JSON parsing failed, trying ast.literal_eval ---")
            # 3. If JSON fails, try ast.literal_eval for Python-like dict strings
            parsed_dict = ast.literal_eval(dict_str)
            print("--- DEBUG (Fact Check): Successfully parsed with ast.literal_eval ---")

        # 4. Standardize the keys and ensure the value is a proper boolean
        is_safe = False
        if 'is_safe_and_correct' in parsed_dict:
            is_safe = parsed_dict['is_safe_and_correct']
        elif 'is_factually_correct' in parsed_dict:
            is_safe = parsed_dict['is_factually_correct']
        
        # 5. Convert string boolean representations to actual booleans
        if isinstance(is_safe, str):
            is_safe = is_safe.lower() in ('true', '1', 'yes')
        
        final_verdict = {
            "is_safe_and_correct": bool(is_safe)  # Only boolean value, no explanation
        }
        
        print(f"--- DEBUG (Fact Check): Parsed verdict ---\n{final_verdict}\n--------------------------------------------------\n")
        return final_verdict
        
    except (ValueError, SyntaxError, json.JSONDecodeError, IndexError, KeyError) as e:
        print(f"--- DEBUG (Fact Check): FAILED TO PARSE RESPONSE. Error: {e} ---")
        print(f"--- Raw text that caused error: {response_text} ---")
        return {"is_safe_and_correct": False}
    # --- END of requested change ---
    
    # # Return default safe response to skip fact-checking (no explanation key)
    # return {"is_safe_and_correct": True}
    
# --- 4. Application UI ---

llm = load_llm_model()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "สวัสดีครับ เราคือเพื่อนใจวัยเรียน AI มีอะไรให้ช่วยรับฟังไหมครับ"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("พิมพ์ข้อความของคุณที่นี่..."):
    if not llm:
        st.warning("Model is not ready. Please try again later.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_stream = get_initial_answer(prompt, llm)
            full_response = st.write_stream(response_stream)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        with st.spinner("กำลังตรวจสอบความปลอดภัยของคำตอบ..."):
            verdict = run_fact_checking_layer(prompt, full_response, llm)
        
        print(f"--- DEBUG (UI): Final verdict received ---\n{verdict}\n------------------------------------------\n")

        if not verdict.get("is_safe_and_correct", True):
            warning_message = (
                f"⚠️ **ข้อควรระวัง:** คำตอบนี้อาจมีความไม่เหมาะสม "
                f"({verdict.get('explanation', 'ไม่มีคำอธิบาย')})"
            )
            st.warning(warning_message)
            st.session_state.messages.append({"role": "assistant", "content": warning_message})

