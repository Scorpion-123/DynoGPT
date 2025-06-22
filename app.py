import streamlit as st
import base64
from PIL import Image
import pickle, os
from PyPDF2 import PdfReader 
from groq import Groq
from langchain.chat_models import init_chat_model

# --------------------------
# Page Config
# --------------------------
st.set_page_config(
    page_title="Dyno GPT",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# Custom Dark Themed CSS Styling
# --------------------------
st.markdown("""
    <style>
    /* Background and font styling */
    body {
        background-color: #121212;
        color: #E0E0E0;
    }

    /* Center title */
    .centered-title {
        text-align: center;
        font-size: 50px;
        font-weight: bold;
        padding-left: 40px;
        color: #14FFF7;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
            
    .centered-sub-title {
        text-align: center; 
        font-weight: bold; 
        color: gray; 
        margin-bottom: 10px;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.6);
    }

    /* Input fields */
    .stTextInput > div > div > input {
        background-color: #1F1F1F;
        color: #FFFFFF;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #202124 !important;
        color: #ffffff;
    }

    /* Button styling */
    .stButton>button {
        background-color: #00ADB5;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }

    .stButton>button:hover {
        background-color: #14FFF7;
        color: black;
    }

    /* Custom divider */
    .divider {
        border: 2px solid #00ADB5;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    </style>
""", unsafe_allow_html=True)


# Convert the Image to base64.
def get_base64_of_image(image_path):
    with open(image_path, 'rb') as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded


# --------------------------
# Sidebar
# --------------------------

st.markdown("""
    <style>
        /* Set sidebar width */
        section[data-testid="stSidebar"] {
            width: 415px !important;        /* sidebar width */
            min-width: 415px !important;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🤖 Choose your LLM Model")
llm_model_name = st.sidebar.selectbox(
    "Select LLM",
    ("gemma2-9b-it", "llama-3.1-8b-instant", "llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b", 
     "compound-beta-mini", "meta-llama/llama-4-maverick-17b-128e-instruct")
)

st.sidebar.success(f" ⚙️ For Image Classification Use LLM Model : \n **meta-llama/llama-4-maverick-17b-128e-instruct**.")


st.sidebar.markdown("### 💬 Enter Your Prompt")
user_prompt = st.sidebar.text_area("Start typing your prompt here...", placeholder="Ask me anything...", height=150)

main_placeholder = st.sidebar.empty()
output_generated = False

# Action Button
prompt_btn = st.sidebar.button("🚀 Run Prompt")
file_upload = st.sidebar.file_uploader("Upload a File", type=['pdf', 'png', 'jpeg', 'jpg'])

if prompt_btn:
    if user_prompt.strip():

        # Fetching API Key.
        with open('secret-key-groq.pkl', 'rb') as file:
            api_key = pickle.load(file)


        if file_upload is not None:
            
            filetype = file_upload.name.strip().split('.')[1]
            print(filetype)

            # Reading and Storing the Contents of the Uploaded file, which will be deleted later after we have used the file.
            with open(file_upload.name, 'wb') as file:
                file.write(file_upload.read())


            # Uploaded file is a PDF. We will process the PDF, extract the text and then we'll add the user_prompt in the text
            # and then we will give the final_prompt to the llm.
            if (filetype == 'pdf'):
                reader = PdfReader(file_upload.name)
                text = ""
                text += f"{user_prompt}\n\n"

                for page in reader.pages:
                    text += page.extract_text()


                llm_model = init_chat_model(api_key=api_key, model=llm_model_name, model_provider='groq')
                output = llm_model.invoke(text).content

                
            # Uploaded File is an Image. We have to create a new designated prompt template for an Image.
            # In order to ask questions about the image we need some special setup.
            else:
                
                try:
                    image_data = get_base64_of_image(file_upload.name)

                    client = Groq(api_key=api_key)
                    comp = client.chat.completions.create(
                        model = llm_model_name,
                        messages = [
                            {
                                'role': 'user',
                                'content': [
                                    {
                                        'type': 'text',
                                        'text': f'{user_prompt}'
                                    },
                                    {
                                        'type': 'image_url',
                                        'image_url': {
                                            'url': f"data:image/jpeg;base64,{image_data}",
                                        }
                                    }
                                ]
                            }
                        ],

                        temperature=0.7,
                        max_completion_tokens=1024,
                        top_p=1,
                        stream=False,
                        stop=None,
                    )

                    output = comp.choices[0].message.content
                    
                
                except Exception as e:
                    st.warning(e)


            # Deleting the File is necessary otherwise, we will ran out of memory.
            # Whether there is a 'pdf' or an 'image' file, it must be deleted from memory after it's use.
            try:
                file_path = os.path.join(os.getcwd(), file_upload.name)

                os.remove(file_path)
                print(f"File '{file_path}' deleted successfully.")

            except FileNotFoundError:
                print(f"File '{file_path}' not found.")
            except PermissionError:
                print(f"Permission denied to delete file '{file_path}'.")
            except Exception as e:
                print(f"An error occurred: {e}")
                    

            output_generated = True


        # This code snippet indicates that we have only recieved a text prompt as code and no image.
        else:
            main_placeholder.text("Processing Your Prompt...✅✅✅")

            # Initializing the LLM Model.
            llm_model = init_chat_model(api_key=api_key, model=llm_model_name, model_provider='groq')
            output = llm_model.invoke(user_prompt).content
            
            output_generated = True

    else:
        st.warning("Please enter a prompt before running.")


# Refresh and opens a new chat as soon as the user presses this button.
if st.sidebar.button("New Chat 💭"):
    with open("current_chat.pkl", 'wb') as file:
        pickle.dump([], file)

    st.rerun()



# --------------------------
# Main UI
# --------------------------
st.markdown('<div class="centered-title">Dyno GPT 🚀</div>', unsafe_allow_html=True)
st.markdown("<div class='centered-sub-title'>🧠 Power your imagination with Dyno GPT – The AI That Works for You</div>", unsafe_allow_html=True)

# Loading the Images in-order to Display.
image1 = get_base64_of_image('images/search.png')
image2 = get_base64_of_image('images/chat-gpt.png')
image3 = get_base64_of_image('images/link.png')
image4 = get_base64_of_image('images/sparkling.png')
image5 = get_base64_of_image('images/llama.png')
image6 = get_base64_of_image('images/langchain.png')
image7 = get_base64_of_image('images/tensorflow.png')


st.markdown(
    """
    <style>
    .icon-row {
        display: flex;
        justify-content: space-around;
        align-items: center;
        gap: 15px;
    }

    .icon-row img {
        width: 40px;
        height: 40px;
        transition: transform 0.2s ease-in-out;
    }

    .icon-row img:hover {
        transform: scale(1.1);
        cursor: pointer;
    }
    </style> """, unsafe_allow_html=True)

st.markdown(
    f"""
                    
    <div class="icon-row">
        <a href="https://www.langchain.com/" target="_blank">
            <img src="data:image/jpeg;base64,{image1}" alt="LangChain">
        </a>
        <a href="https://www.langgraph.dev/" target="_blank">
            <img src="data:image/jpeg;base64,{image2}" alt="LangGraph">
        </a>
        <a href="https://www.google.com/" target="_blank">
            <img src="data:image/jpeg;base64,{image3}" alt="Google">
        </a>
        <a href="https://www.google.com/" target="_blank">
            <img src="data:image/jpeg;base64,{image4}" alt="Google">
        </a>
        <a href="https://www.google.com/" target="_blank">
            <img src="data:image/jpeg;base64,{image5}" alt="Google">
        </a>
        <a href="https://www.google.com/" target="_blank">
            <img src="data:image/jpeg;base64,{image6}" alt="Google">
        </a>
        <a href="https://www.google.com/" target="_blank">
            <img src="data:image/jpeg;base64,{image7}" alt="Google">
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# If prompt result is generated in the current state, then we will display the current result and then we will display all 
# the other prompts that we saved previously.
if (output_generated):

    # Displaying the Current Answer for the Prompt.
    st.markdown("<hr style='border: 1px solid #00ADB5;'>", unsafe_allow_html=True)
    st.markdown("### 🔍 Output")
    st.info(f"Simulated response using model '**{llm_model_name}**'\n\n> _{output}_")


    # Loading the previous prompts of the same chat.
    with open('current_chat.pkl', 'rb') as current_chat:
        current_chat = pickle.load(current_chat)


    st.markdown("<hr style='border: 1px solid #00ADB5;'>", unsafe_allow_html=True)
    st.markdown("### 🔍Previous Prompts.")
    # Itterate and Display the previous prompts.
    if (len(current_chat) != 0):

        for entry in current_chat:
            st.markdown(f"""
                <div style="background-color:#1f1f2e; padding:20px; border-radius:10px; margin-bottom:20px; box-shadow: 0 4px 10px rgba(0,0,0,0.2);">
                    <p style="color:#AAAAAA; margin:0px; font-size:20px;">❓ Prompt</p>
                    <h4 style="color:#00ADB5; margin-top:0px;">{entry['question']}</h4>

                    LLM Model: {entry['model']}

                    ✅ Answer:
                {entry['content']}

                """, unsafe_allow_html=True)

    # Add the prompt to the current chat.
    current_chat.insert(0, {
        "question": user_prompt,
        "model": llm_model_name,
        "content": output
    })


    # Modify and Update the prompt in the chat.
    with open('current_chat.pkl', 'wb') as file:
        pickle.dump(current_chat, file)


# As per the current state if the user has not given current prompt then it will only display the previous prompts.
else:

    # Loading the previous prompts of the same chat.
    with open('current_chat.pkl', 'rb') as current_chat:
        current_chat = pickle.load(current_chat)


    # Itterate and Display the previous prompts.
    if (len(current_chat) != 0):

        st.markdown("<hr style='border: 1px solid #00ADB5;'>", unsafe_allow_html=True)
        st.markdown("### 🔍Previous Prompts.")

        for entry in current_chat:
            st.markdown(f"""
                <div style="background-color:#1f1f2e; padding:20px; border-radius:10px; margin-bottom:20px; box-shadow: 0 4px 10px rgba(0,0,0,0.2);">
                    <p style="color:#AAAAAA; margin:0px; font-size:20px;">❓ Prompt</p>
                    <h4 style="color:#00ADB5; margin-top:0px;">{entry['question']}</h4>

                    🌐 LLM Model: {entry['model']}
                    
                    ✅ Answer:
                    {entry['content']}_
                """, unsafe_allow_html=True)
            

# Footer
st.markdown("<hr style='border: 1px solid #00ADB5;'>", unsafe_allow_html=True)
st.markdown("🔧 Developed with ❤️ by Dyno Team | 🕶️ Dark Mode Inspired UI", unsafe_allow_html=True)
