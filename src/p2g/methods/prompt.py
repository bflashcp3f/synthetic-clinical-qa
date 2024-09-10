
import os
import json
import time
import openai
import backoff 
import asyncio
import tiktoken
import torch
import transformers

from tqdm import tqdm

from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer

from p2g.tasks.base import Task, DATA_PATH
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI

CHATCOMPLETION_MODEL = [
    'gpt-4o', 'gpt-4o-2024-05-13',
    'gpt-4-turbo', 'gpt-4-turbo-2024-04-09', 'gpt-4', 
    'gpt-3.5-turbo', 'gpt-35-turbo-0613', 'gpt-35-turbo-16k', 'gpt35-turbo'
]

AZURE_MODELS = {
    'gpt-35-turbo-0613': 'gpt-3.5-turbo',
    'gpt-35-turbo-16k': 'gpt-3.5-turbo-16k',
    'gpt-35-turbo-0613': 'gpt-3.5-turbo-0613',
    'gpt-4o-2024-05-13': 'gpt-4o-2024-05-13',
}

OPEN_SOURCE_MODELS = {
    'tablellama': 'osunlp/TableLlama',
    'llama2-chat-7b': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2-chat-13b': 'meta-llama/Llama-2-13b-chat-hf',
    'llama3-8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'codellama-7b-instruct': 'codellama/CodeLlama-7b-Instruct-hf',
    'codellama-13b-instruct': 'codellama/CodeLlama-13b-Instruct-hf',
    'codellama-34b-instruct': 'codellama/CodeLlama-34b-Instruct-hf',
    'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.1',
}

client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

aclient = AsyncOpenAI(
    api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

client_azure = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2023-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

aclient_azure = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def num_tokens_from_string(string: str, encoding, open_source=False) -> int:
    """Returns the number of tokens in a text string."""

    if not open_source:
        num_tokens = len(encoding.encode(string))
    else:
        num_tokens = len(encoding.encode(string, return_tensors='pt')[0])
    return num_tokens


def prompt_os(messages_list, pipeline, temp_max, temp_min, top_p, max_tokens, batch_size) -> list:

    response_list = []
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    temp_slice = (temp_max - temp_min) / len(messages_list)
    assert temp_slice >= 0
    
    # for prompt in prompt_list:
    for i in tqdm(range(0, len(messages_list), batch_size)):
        
        temperature = temp_min + temp_slice * i
        temperature = float(f"{temperature:.2f}")
        if temp_slice > 0:
            print(f"Running batch {i} to {i + min(batch_size, len(messages_list)-i)} with temperature {temperature}")
        
        messages_batch = messages_list[i : i + batch_size]
        prompt_batch = [pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
        # breakpoint()

        try:

            start_time = time.time()
            outputs = pipeline(
                prompt_batch,
                max_new_tokens=max_tokens,
                pad_token_id=pipeline.tokenizer.eos_token_id,
                eos_token_id=terminators,
                temperature=temperature if temperature>0 else 0.0001,
                top_p=top_p,
                num_return_sequences=1,
            )
            end_time = time.time()
            # print(f'Finish prompting in {end_time - start_time} seconds')
            
            response_batch = [output[0]['generated_text'][len(prompt):] for output, prompt in zip(outputs, prompt_batch)]

            response_list += response_batch
            # breakpoint()
                
        except Exception as e:
            print(e)
            raise e

    return response_list

@backoff.on_exception(backoff.expo, openai.OpenAIError)
def chatcompletions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

@backoff.on_exception(backoff.expo, openai.OpenAIError)
def chatcompletions_with_backoff_azure(**kwargs):
    return client_azure.chat.completions.create(**kwargs)

def prompt_sync(messages_list, api_source, sleep_time, model, temp_max, temp_min, top_p, max_tokens, stop) -> list:
    
    response_list = []    
    temp_slice = (temp_max - temp_min) / len(messages_list)
    assert temp_slice >= 0
    
    for i, messages in enumerate(messages_list):
        
        temperature = temp_min + temp_slice * i
        temperature = float(f"{temperature:.2f}")
        
        if model in CHATCOMPLETION_MODEL:
            
            if api_source == 'openai':
                response = chatcompletions_with_backoff(model=model, messages=messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop)
            elif api_source == 'azure':
                response = chatcompletions_with_backoff_azure(model=model, messages=messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop)
            else:
                raise ValueError(f"api_source {api_source} not supported")
            response = response.choices[0].message.content
        else:
            raise ValueError(f"model {model} not supported")
        response_list.append(response)

        # Sleep to avoid hitting the API rate limit
        print(f"Sleep for {sleep_time} seconds")
        time.sleep(sleep_time)
        
    return response_list


async def dispatch_chatcompletion_requests(
    api_source: str,
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop: str,
) -> list[str]:
    # Dispatches requests to OpenAI API asynchronously.
    
    if api_source == 'openai':
        async_responses = [
            aclient.chat.completions.create(
                model=model,
                messages=x,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
            )
            for x in messages_list
        ]
        return await asyncio.gather(*async_responses)
    elif api_source == 'azure':
        async_responses = [
            aclient_azure.chat.completions.create(
                model=model,
                messages=x,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
            )
            for x in messages_list
        ]
        return await asyncio.gather(*async_responses)
    else:
        raise ValueError(f"api_source {api_source} not supported")


def prompt_async(messages_list, api_source, sleep_time, model, temp_max, temp_min, top_p, max_tokens, stop, batch_size) -> list:

    response_list = []
    temp_slice = (temp_max - temp_min) / len(messages_list)
    
    for i in tqdm(range(0, len(messages_list), batch_size)):
        
        temperature = temp_min + temp_slice * i
        temperature = float(f"{temperature:.2f}")
        if temp_slice > 0:
            print(f"Running batch {i} to {i + min(batch_size, len(messages_list)-i)} with temperature {temperature}")
        
        messages_batch = messages_list[i : i + batch_size]

        if model in CHATCOMPLETION_MODEL:

            try:
                response_list_batch = asyncio.run(
                    dispatch_chatcompletion_requests(
                        api_source,
                        messages_batch,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        stop=stop,
                    )
                )
                response_list_batch_no_content = [item for item in response_list_batch if "content" not in vars(item.choices[0].message)]
                if len(response_list_batch_no_content) > 0:
                    for item in response_list_batch_no_content:
                        print(item)
                        
                response_list_batch = [item.choices[0].message.content if "content" in vars(item.choices[0].message) else "" for item in response_list_batch]
                        
            except Exception as e:
                print(e)
                
                response_list_batch = ["No output due to Azure OpenAI's content management." for _ in range(len(messages_batch))]
            
        else:
            raise ValueError(f"model {model} not supported")

        response_list.extend(response_list_batch)

        # Sleep to avoid hitting the API rate limit
        print(f"Sleep for {sleep_time} seconds")
        time.sleep(sleep_time)

    return response_list


def prompt_naive(args, task, open_source=False):
    
    if not open_source:
        if args.api_source == 'openai':
            encoding = tiktoken.encoding_for_model(args.backend)
            max_length = 4000 if 'gpt-3.5' in args.backend else 8000
        else:
            encoding = tiktoken.encoding_for_model(AZURE_MODELS[args.backend])
            max_length = 4000 if 'gpt-3.5' in AZURE_MODELS[args.backend] else 8000
    else:
        if args.backend not in OPEN_SOURCE_MODELS:
            raise ValueError(f"model {args.backend} not supported")
    
        model_name = OPEN_SOURCE_MODELS[args.backend]
        encoding = AutoTokenizer.from_pretrained(model_name)
        encoding.pad_token = encoding.eos_token
        max_length = 8000 if encoding.model_max_length > 8000 else encoding.model_max_length
        
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            # model_kwargs= {
            #     "device_map": "auto", 
            #     "load_in_4bit": True,
            #     "bnb_4bit_compute_dtype": torch.bfloat16
            # }
        )
    
    messages_list = task.get_input_list()
    # print(messages_list[0][0]['content'])
    # breakpoint()
        
    if not open_source:
        if not args.async_prompt:
            response_list = prompt_sync(messages_list, args.api_source, args.sleep_time, args.backend, args.temp_max, args.temp_min, args.top_p, args.max_tokens, args.stop)
        else:
            response_list = prompt_async(messages_list, args.api_source, args.sleep_time, args.backend, args.temp_max, args.temp_min, args.top_p, args.max_tokens, args.stop, args.batch_size)
    else:
        response_list = prompt_os(messages_list, pipeline, args.temp_max, args.temp_min, args.top_p, args.max_tokens, args.batch_size)
    # breakpoint()

    # Update the task with the response
    task.update_output(response_list)
    # breakpoint()

    # Save the task output
    task.save_output()


def build_icl_promtpt(task, prompt_list, icl_examples, icl_num=1):
    """
    Build the prompt with icl examples
    """

    icl_examples_str = task.build_icl_examples_str(icl_examples, icl_num)
    print(icl_examples_str)

    icl_prompt_list = []
    for prompt in prompt_list:
        icl_prompt = prompt.replace('{{icl_examples}}', icl_examples_str)
        icl_prompt_list.append(icl_prompt)
        print(icl_prompt, '\n\n\n')
        
    return icl_prompt_list


def prompt_icl(args, task, open_source=False):

    if not open_source:
        if args.api_source == 'openai':
            encoding = tiktoken.encoding_for_model(args.backend)
            max_length = 4000 if 'gpt-3.5' in args.backend else 8000
        else:
            encoding = tiktoken.encoding_for_model(AZURE_MODELS[args.backend])
            max_length = 4000 if 'gpt-3.5' in AZURE_MODELS[args.backend] else 8000
    else:
        model_name = OPEN_SOURCE_MODELS[args.backend]
        encoding = AutoTokenizer.from_pretrained(model_name)
        encoding.pad_token = encoding.eos_token
        max_length = 8000 if encoding.model_max_length > 8000 else encoding.model_max_length

        model = transformers.pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    prompt_list = task.get_input_list()
    # print(prompt_list[0])
    # breakpoint()

    if not open_source:
        if not args.async_prompt:
            response_list = prompt_sync(prompt_list, args.api_source, args.sleep_time, args.backend, args.temperature, args.top_p, args.max_tokens, args.stop)
        else:
            response_list = prompt_async(prompt_list, args.api_source, args.sleep_time, args.backend, args.temperature, args.top_p, args.max_tokens, args.stop, args.batch_size)
    else:
        response_list = prompt_os(prompt_list, model, args.temperature, args.top_p, args.max_tokens, args.batch_size)

    # Update the task with the response
    task.update_output_icl(response_list)
    # print(task.get_output_list()[0])
    # breakpoint()

    # Save the task output
    task.save_output_icl()
    # breakpoint()

