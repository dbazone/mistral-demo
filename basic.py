from huggingface_hub import snapshot_download
from pathlib import Path
from mistral_inference.model import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import os
import time
from loggerSetup import setupLogger

# Setup logger
logger = setupLogger("mistral")


def invoke(prompt):
    start_time = time.time()  # Start the timer
    accessToken = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    mistralModelsPath = Path.home().joinpath('mistral_models', '7B-Instruct-v0.3')  
    mistralModelsPath.mkdir(parents=True, exist_ok=True)

    logger.info("reuse/download model..")
    snapshot_download(repo_id="mistralai/Mistral-7B-Instruct-v0.3", allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], local_dir=mistralModelsPath, token=accessToken)

    tokenizer = MistralTokenizer.from_file(f"{mistralModelsPath}/tokenizer.model.v3")
    model = Transformer.from_folder(mistralModelsPath)

    logger.info("preparing input")
    completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
    tokens = tokenizer.encode_chat_completion(completion_request).tokens

    logger.info("generating response")
    out_tokens, _ = generate([tokens], model, max_tokens=4096, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id )

    result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
    print(result)
    end_time = time.time()  # End the timer
    logger.info(f"Response took {end_time - start_time:.6f} seconds to execute")



if __name__ == "__main__":
    logger.info("application started")

    prompt = """I would like you to build the story with theme of 'Adventures of superman in India'. 
    The story should be targeted for teenagers and should have elements of engagement from both boys and girls as target audience. 

    The story timeline should be starting back to 3000 years back to present day and should include the encounter of superman with famous historical and current personalities including Gautam Budha, Ashoka, Chankya Guru Nanak, Shivaji Maharaj, Viveknanda, APJ Abdul kalam. 

    The story should be based on Superman perspective
    """
    invoke(prompt)
    logger.info("application finished")
