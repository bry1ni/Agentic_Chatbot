from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage

# from huggingface_hub import login
# login() # generate HF Key

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

model = ChatHuggingFace(llm=llm, verbose=True)

query = HumanMessage(content="what do you know about machine learning ?")

response = model.invoke([query])

print(response.content)