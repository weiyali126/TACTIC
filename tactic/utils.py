import argparse
import ast
import json
import os
import sys
import re
import subprocess
import numpy as np
import spacy
from tqdm import tqdm
import yaml
import time
import json_repair
from datetime import datetime
from pathlib import Path
from loguru import logger
from lxml import etree
from typing import Dict, Iterable, List, Union, Tuple
from mosestokenizer import MosesDetokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tenacity import Retrying, stop_after_attempt, wait_random_exponential
from dotenv import load_dotenv, find_dotenv 
from tactic.config import RETRY_THRESHOLD, LOG_DIR, FEW_SHOT_PATH

load_dotenv(find_dotenv())


def setup_logger(configs, output_path=LOG_DIR):
    log_dir = output_path
    os.makedirs(log_dir, exist_ok=True)
    logger.remove()
    logger.add(
        os.path.join(log_dir, "log_{time:YYYY-MM-DD}.log"),
        colorize=True,
        level='INFO',
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}.{function} | {message}",
        rotation=("00:00"), 
        retention='30 days',
        encoding="utf-8"
    )
    logger.add(
        sys.stderr,  
        colorize=True,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {level: <8} | {module}.{function} | {message}",
         
    )
    return 

def get_dataset(dataset, lp):
    '''
    Load dataset
    '''
    data_path = f"{DATA_PATH}/{dataset}.{lp}/test.jsonl"
    data = []
    print(f"Loading data from {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f: 
        data = [json.loads(line) for line in f]
    return data

def extract_json(text):
    """
    Use the stack to match multiple layers of nested JSON
    """
    stack = []
    json_text = ""
    inside_json = False  # Whether to enter the JSON structure

    for i, char in enumerate(text):
        if char == "{":
            stack.append(char)
            if not inside_json:
                inside_json = True  # Enter JSON 
                json_text = ""  # Empty the string and start recording JSON
        if inside_json:
            json_text += char  # Record JSON content
        if char == "}":
            stack.pop()
            if not stack:  # If the stack is empty, complete JSON ends
                break

    return json_text
    
def extract_json_with_step(agent, prompt, response_format=None, tag=None):
    """
    Extracts JSON content from agent response and ensures all specified tags exist.

    Parameters:
        agent: The agent used to generate responses.
        prompt: The input prompt to the agent.
        tag (str | list): The tag or list of tags to validate in the JSON response.

    Returns:
        dict: Extracted JSON content if all tags exist.

    Raises:
        ValueError: If extraction fails after maximum retries.
    """
    tags = [tag] if isinstance(tag, str) else tag 

    retry_cnt = 1
    while retry_cnt <= RETRY_THRESHOLD:
        response = None
        try:
            agent.init_messages()
            response = agent.step(prompt).msgs[0].content.strip()
            
            if response_format and tag:
                response = extract_json(response)
                response = json_repair.loads(response)
            
            # Compute assistant token
            context, token_count = agent.memory.get_context()
            response_token = agent.model_backend.token_counter.count_tokens_from_messages([context[-1]])
            max_tokens = agent.model_backend.token_limit
            
            if response_token + 500 > max_tokens:
                logger.warning(f"Response token reached the maximum: {response_token}/{max_tokens} | Response: {response} | Prompt: {prompt}")

                raise ValueError(f"Response token reached the maximum: {response_token}/{max_tokens}")
                
            if not tag or (tag and all(t in response for t in tags)):
                return response

            logger.warning(f"[Retry {retry_cnt}/{RETRY_THRESHOLD}] Response not return | Response: {response} | Prompt: {prompt}")
                
        except Exception as e:
            logger.warning(f"[Retry {retry_cnt}/{RETRY_THRESHOLD}] Failed to Extract JSON: {e} | Response: {response} | Prompt: {prompt}")
        
        retry_cnt += 1

    logger.error(f"Failed to extract JSON, retries maximum reached: {retry_cnt}")
    raise ValueError(f"Failed to extract JSON, retries maximum reached: {retry_cnt}")

def extract_html_with_step(agent, prompt, tag="translation"):
    pattern = rf"<{tag}>(.*?)</{tag}>"
    retry_cnt = 1
    while retry_cnt <= RETRY_THRESHOLD:
        response = None
        try:
            agent.init_messages()
            response = agent.step(prompt).msgs[0].content
            answer = response.split('</think>')[1] if '</think>' in response else response
            match = re.search(pattern, answer, re.DOTALL)
            
            if match: 
                translation = match.group(1).strip()
                return translation
                
        except Exception as e:
            logger.info(f"[Retry {retry_cnt}/{RETRY_THRESHOLD}] Failed to Extract HTML: {e} | Response: {response} | Prompt: {prompt}")
        
        retry_cnt += 1

    logger.error(f"Failed to extract html, retries maximum reached: {retry_cnt}")
    raise ValueError(f"Failed to extract from html, retries maximum reached: {retry_cnt}")
        
def extract_html(raw, tag="translation"):
    match = re.search(rf"<{tag}>(.*?)</{tag}>", raw, re.DOTALL)
    if match:
        text = match.group(1).strip()
        return text
    else:
        raise ValueError(f"extract {tag} from html error, raw: {raw}")

LANG_TABLE = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "az": "Azerbaijani",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "ko": "Korean",
    "ky": "Kyrgyz",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mk": "Macedonian",
    "mr": "Marathi",
    "ms": "Malay",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sr": "Serbian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


NLLB_CODE = {
    "cs": "ces_Latn",
    "de": "deu_Latn",
    "en": "eng_Latn",
    "is": "isl_Latn",
    "ja": "jpn_Jpan",
    "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl",
    "zh": "zho_Hans",
}

ISO1_ISO3_map = {
    "cs": "ces",
    "de": "deu",
    "en": "eng",
    "is": "isl",
    "ja": "jpn",
    "ru": "rus",
    "uk": "ukr",
    "zh": "zho",
}


# -------tower_eval utils.py----------
PathInput = Union[str, Path]
PATTERN_OPEN_TAG = r"<\s*([\w]+)\s*>\s*"
PATTERN_CLOSE_TAG = r"\s*<\s*\/\s*([\w]+)\s*>"
PATTERN_SHOT_NAME = r"^([0-9]+)[-_]shot.*$"

SPACY_LANGUAGE_TO_MODEL = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "nl": "nl_core_news_sm",
    "pt": "pt_core_news_sm",
    "ru": "ru_core_news_sm",
    "it": "it_core_news_sm",
    "pl": "pl_core_news_sm",
    "es": "es_core_news_sm",
    "sv": "sv_core_news_sm",
    "zh": "zh_core_web_sm",
    "ko": "ko_core_news_sm",
}

def assess_progress(
    input_lines: List[str],
    output_file: str,
    metadata: dict,
    metadata_file: str,
    overwrite_generations: bool = False,
) -> Tuple[List[str], List[str], dict, int, int]:
    """ 
    Assess progress, resuming from last line generated
    """
    total_lines = len(input_lines)
    if os.path.exists(output_file) and not overwrite_generations:
        processed_lines = read_lines(output_file, unescape_newline=True)
        num_processed_lines = get_num_processed_lines(output_file)
    else:
        processed_lines = []
        num_processed_lines = 0
    # We assume that if the metadata file exists it already contains the information of the generation times.
    # If it doesn't exist, then the metadata will be the config of the task and we will add the generation_time field to it
    if os.path.exists(metadata_file) and not overwrite_generations:
        metadata = load_json_file(metadata_file)
    else:
        metadata["generation_time"] = []

    assert (
        num_processed_lines <= total_lines
    ), f"MORE PROCESSED LINES ({num_processed_lines}) THAN INPUT LINES ({total_lines})!"
    # Skip the lines already processed
    input_lines = input_lines[num_processed_lines:]

    return input_lines, processed_lines, metadata, num_processed_lines, total_lines


def args_to_dict(args, prefix: str, strip_prefix: bool = False):
    """Filters argparse's `Namespace` into dictionary with arguments
    beginning with the given prefix."""
    prefix += "_"
    d = {}
    for k, v in args.__dict__.items():
        if k.startswith(prefix) or k in ["language", "tokenize", "lowercase"]:
            k = k.replace(prefix, "") if strip_prefix else k
            d[k] = v
    return d


def detokenize(lines: list[str]) -> list[str]:
    detokenizer = TreebankWordDetokenizer()
    detokenized_lines = [detokenizer.detokenize(line.split()) for line in lines]
    return detokenized_lines


def detokenize_moses(lines: list[str], lang: str) -> list[str]:
    detokenizer = MosesDetokenizer(lang)
    detokenized_lines = [detokenizer(line.split()) for line in lines]
    return detokenized_lines


def tokenize_spacy(lines: list[str], language, keep_xml=False) -> list[str]:
    spacy_model = SPACY_LANGUAGE_TO_MODEL[language]
    try:
        tokenizer = spacy.load(spacy_model, disable=["tagger", "parser", "ner"])
    except OSError:
        logger.warning(f"Spacy model {spacy_model} not found. Trying to download...")
        download_proc = subprocess.run(
            ["python", "-m", "spacy", "download", spacy_model]
        )
        tokenizer = spacy.load(spacy_model, disable=["tagger", "parser", "ner"])

    tokenized = list(tokenizer.pipe(lines))
    tokenized = [[token.text for token in sentence] for sentence in tokenized]
    output_lines = [" ".join(sentence) for sentence in tokenized]
    if keep_xml:
        output_lines = [
            re.sub(PATTERN_OPEN_TAG, r"<\1>", output_line)
            for output_line in output_lines
        ]
        output_lines = [
            re.sub(PATTERN_CLOSE_TAG, r"</\1>", output_line)
            for output_line in output_lines
        ]
    return output_lines


def parse_yaml_config(config_path):
    with open(config_path, "r") as config_file:
        config_data = yaml.safe_load(config_file)
    return config_data


def get_task_args(args: dict) -> dict:
    source = args.get("source")
    hypothesis = args.get("hypothesis")
    references = args.get("references")

    task_args = {
        "hypothesis": hypothesis,
        "references": references,
    }
    if source:
        task_args.update({"source": source})
    return task_args


def load_json_file(
    file_path: PathInput,
) -> list:
    """
    Function to to load json files.

    :param file_path: either the handler to the file storing the hypotheses.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_file(
    file_path: PathInput,
) -> list:
    """
    Function to parse error span jsonl files.
    All inputs will be returned as list of dictionaries.

    :param file_path: either the handler to the file storing the hypotheses.
        Must be a jsonl file where each line is a dict which has at least the following keys:
        {"mt": <mt>, "annotations": [{"start": <start>, "end": <end>, "severity": <severity>}, ...]}
    :return:
        -  List of dictionaries
    """
    with open(file_path, "r", encoding="utf-8") as f:
        json_list = list(f)

    return [json.loads(l) for l in json_list]


def list_to_dict(l: list[dict[str, str]]) -> dict[str, list[str]]:
    """
    Assumes lists in dict have the same length.
    """
    d = {}
    for k in l[0].keys():
        d[k] = [d[k] for d in l]
    return d


def dict_to_records(d: dict[str, list[str]]) -> list[dict[str, str]]:
    """
    Assumes lists in dict have the same length.
    """
    records = []
    for i in range(len(list(d.values())[0])):
        records.append({k: v[i] for k, v in d.items()})

    return records


def write_lines(
    path: PathInput,
    lines: Iterable[str],
    escape_newline: bool = False,
    escape_return_char: bool = True,
    verbose: bool = True,
) -> None:
    """Writes lines to a file.

    Lines can be escaped, meaning \n is transformed to \\n.

    Args:
        path: The path to the file.
        lines: The lines to write.
        escape_newline: Whether to escape newlines.
    """
    # make dir, if not exists
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out_lines = []
    for i, line in enumerate(lines):
        if escape_return_char:
            if "\r" in line and verbose:
                logger.opt(ansi=True).warning(
                    f"Detected carriage return in line {i + 1} (\\r). This may cause errors downstream. Escaping. This behaviour is the default; you can turn it off with escape_return_char."
                )
            line = line.replace("\r", "\\r")
        if escape_newline:
            if "\n" in line and verbose:
                logger.opt(ansi=True).warning(
                    f"Found new line in line {i + 1} (\\n). This may cause errors downstream. Escaping."
                )
            line = line.replace("\n", "\\n")
        out_lines.append(line)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines((f"{l}\n" for l in out_lines))


def combine_metrics_args(task_metric: dict, subtask_metric: dict) -> dict:
    eval_metric = task_metric.copy()
    for metric_arg, metric_arg_value in subtask_metric.items():
        eval_metric[metric_arg] = metric_arg_value
    return eval_metric


def generate_with_retries(
    retry_function,
    model_args,
    retry_max_attempts=1,
    retry_multiplier=1,
    retry_max_interval=10,
    retry_min_interval=4,
):
    retryer = Retrying(
        stop=stop_after_attempt(retry_max_attempts),
        wait=wait_random_exponential(
            multiplier=retry_multiplier, max=retry_max_interval, min=retry_min_interval
        ),
        reraise=True,
    )
    return retryer(retry_function, **model_args)


def sample_strings_from_list(strings: list[str], num_samples: int) -> list[str]:
    """ """
    return np.random.choice(strings, num_samples).tolist()


def read_lines(path: PathInput, unescape_newline: bool = False) -> List[str]:
    """Reads lines from a file.
    Lines can be unescapped, meaning \\n is transformed to \n.
    Args:
        path: The path to the file.
        unescape_newline: Whether to unescape newlines.
    Returns:
        The lines in the file."""
    with open(path, encoding="utf-8") as f:
        lines = [l[:-1] for l in f.readlines()]
    if unescape_newline:
        lines = [l.replace("\\n", "\n") for l in lines]
    return lines

def read_gen_data(path: PathInput, unescape_newline: bool = False) -> List[str]:
    """Reads generate data from a file.
    Lines can be unescapped, meaning \\n is transformed to \n.
    Args:
        path: The path to the file.
        unescape_newline: Whether to unescape newlines.
    Returns:
        The lines in the file."""
    with open(path, encoding="utf-8") as f:
        lines = [json.loads(l)['src'] for l in f.readlines()]
    if unescape_newline:
        lines = [l.replace("\\n", "\n") for l in lines]
    return lines

def make_dir_if_not_exists(output_file):
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def log_response(response: str, step: int, lim: int = 10) -> None:
    """
    Logs the response from the model.
    """
    if step < lim:
        if "<" in response:
            response = response.replace("<", "\\<")
            logger.opt(colors=True).info(
                f"<red> Escaped tags in below response to not throw colorizing error. </red>"
            )
        logger.opt(colors=True).info(f"Response {step}: {response.strip()}")


def get_num_processed_lines(output_file: str, **kwargs):
    """
    This function is used to write the output from the last line of the input file.
    """
    if os.path.exists(output_file):
        processed_lines = read_lines(output_file, unescape_newline=False)
        num_processed_lines = len(processed_lines)
        if num_processed_lines > 0:
            logger.opt(ansi=True).warning(
                f"<red>Resuming generation from line {num_processed_lines}.</red>"
            )
    else:
        num_processed_lines = 0

    return num_processed_lines


def load_data_to_records(
    data_path: str,
) -> list[dict[str, str]]:
    """ """
    # prompt data file can either be a jsonl or a csv
    data = load_jsonl_file(data_path)

    return data


def save_to_json(
    save_location: PathInput,
    data: dict,
) -> None:
    """ """
    if isinstance(save_location, str):
        save_location = Path(save_location)
    save_location.parent.mkdir(parents=True, exist_ok=True)
    with open(save_location, "w") as output_path:
        json.dump(data, output_path, indent=4)


def get_sacrebleu_segment_scores(
    hypotheses: list[str],
    references,
    method,
) -> list[float]:
    """ """
    segment_scores = []
    for h, r in zip(hypotheses, references):
        segment_score = method.sentence_score(h, r)
        segment_scores.append(segment_score.score)
    return segment_scores


def get_eval_args_given_task(
    eval_args: Dict[str, str],
    task_name: str,
    subtask: str,
    gen_output_dir: Path,
    eval_data_dir: Path,
    eval_output_dir: Path,
    model_platform: str,
    model_name: str,
    save_name: str,
    model_platform_2: str = None,
    model_name_2: str = None,
) -> Dict[str, str]:
    """ """
    gen_output_path = (gen_output_dir / task_name / subtask / model_platform / save_name / "generation.txt")
    eval_output_path = (eval_output_dir/ task_name / subtask / model_platform / save_name / "evaluation.json")

    eval_data_path = eval_data_dir / subtask / "test.jsonl"
    # add language argument to eval_args as it is needed in some metrics
    lp = subtask.split(".")[1]
    src_lang, trg_lang = get_langs(lp)
    eval_args["lp"] = {"src_lang": src_lang, "trg_lang": trg_lang}

    return gen_output_path, eval_data_path, eval_output_path, eval_args


def text_to_label(
    text: str,
    text_type: str,
    label_space: List[str] = None,
    return_is_random: bool = False,
) -> int:
    is_random = False
    if text_type == "categorical":
        label = int(text.strip())
        is_random = False
    # if hypothesis is text
    elif text_type == "text":
        label = np.random.randint(len(label_space))
        is_random = True
        for l in label_space:
            if re.search(l, text, re.IGNORECASE) is not None:
                label = label_space.index(l)
                is_random = False
                break
    if return_is_random:
        return label, is_random
    else:
        return label


def handle_subprocess(subprocess_args: List[str], check_output: bool = False):
    output = 1
    if check_output:
        process = subprocess.Popen(
            subprocess_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        output = ""

        # Read and print the output in real-time
        for line in iter(process.stdout.readline, ""):
            print(line, end="")  # Print each line as it's received
            output += line  # Append each line to the output variable
    else:
        try:
            process = subprocess.Popen(subprocess_args)
            while process.poll() is None:
                time.sleep(1)
            logger.info("Generation process has finished.")
            output = 0
        except KeyboardInterrupt:
            process.terminate()
    return output


def tokenize_text(references, tokenizer):
    tokenized_prompts = []
    logger.info(f"Tokenizing text...")
    for reference in tqdm(references):
        tokenized_prompts.append(tokenizer.encode(reference))
    return tokenized_prompts


def get_langs(lp):
    # TODO: This has to be updated to account for more languages
    valid_langs = [
        "de",
        "en-us",
        "en-gb",
        "en-uk",
        "en",
        "es-latam",
        "es",
        "fr",
        "fr-ca",
        "fr-fr",
        "it",
        "it-it",
        "ko",
        "ko-ko",
        "nl",
        "nl-nl",
        "pl",
        "pl-pl",
        "pt-br",
        "pt",
        "pt-pt",
        "ru",
        "zh-CN",
        "zh-cn",
        "zh-TW",
        "zh-tw",
        "zh",
        "ja",
        "hi",
        "uk",
        "cs",
        "is",
        "sv",
        "sv-se",
        "ar",
        "tr",
        "el",
        "hu",
        "hu-hu",
        "ro",
        "ro-ro",
        "fi",
        "fi-fi",
        "sk",
        "da",
        "da-da",
        "lt",
        "lv",
        "sl",
        "et",
        "bg",
        "no",
        "no-no",
        "ca",
        "hr",
        "ga",
        "mt",
        "gl",
    ]
    lang_pattern = "|".join(valid_langs)
    lp_pattern = rf"^({lang_pattern})-({lang_pattern})$"
    match = re.match(lp_pattern, lp)
    if match:
        # We have a language pair, so both src_lang and trg_lang will be extracted from lp
        src_lang = match.group(1)
        trg_lang = match.group(2)
        return src_lang, trg_lang
    elif lp in valid_langs:
        # the task is monolingual, hence we only have one language. So, we set the src_lang only. trg_lang will be set to None
        src_lang = lp
        trg_lang = None
        return src_lang, trg_lang
    return None, None


def add_average_generation_time(
    input_file: str, metadata_file: str, language: str, mode: str = "lps"
):
    """
    Given the generation file, the metadata file and the mode it calculates the time to generate the outputs.
    Depending on the mode, it can be either the lines-per-second (lps) or words-per-second (wps)

    Args:
        input_file (str): The generation file
        metadata_file (str): The metadata file that contains the generation_time which is a list
        mode (str, optional): The mode for calculating the time. It can be either the lines-per-second (lps) or words-per-second (wps). Defaults to "lps".
    """
    lines = read_lines(input_file)
    metadata = load_json_file(metadata_file)
    generation_time = metadata.get("generation_time")
    if generation_time:
        total_time = sum(generation_time)
        if mode == "lps":
            average_time = len(lines) / total_time
        # TODO: Add the support of wps mode
    else:
        logger.error(
            f"The generation_time information doesn't exist in the {metadata_file}. We are going to set the average generation time to -1."
        )
        average_time = -1
    metadata[f"generation_time_average ({mode})"] = average_time
    save_to_json(metadata_file, metadata)


def parse_dict_arg(arg):
    try:
        # Safely evaluate the string as a Python expression
        parsed = ast.literal_eval(arg)
        if isinstance(parsed, dict):
            return parsed
        else:
            raise argparse.ArgumentTypeError("Argument is not a valid dictionary")
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Argument is not a valid dictionary")


def write_jsonl(path, data):
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def match_xml(string_1: etree._Element, string_2: etree._Element):
    if string_1.tag != string_2.tag:
        return False

    string_1 = list(string_1.iterchildren())
    string_2 = list(string_2.iterchildren())

    if len(string_1) != len(string_2):
        return False

    for s1, s2 in zip(string_1, string_2):
        if not match_xml(s1, s2):
            return False
    return True


def split_by_xml_tags(string: str):
    try:
        root = etree.fromstring(f"<root>{string}</root>")
        parts = []

        for element in root.iter():
            if element.tag == "root":
                if element.text:
                    parts.append(element.text)
                continue

            if element.text:
                parts.append(element.text)
            if element.tail:
                parts.append(element.tail)

        return [p.strip() for p in parts if p.strip()]
    except etree.XMLSyntaxError:
        return [string]


def split_pairs_by_xml_tags(string1: str, string2: str):
    try:
        string1_xml = etree.fromstring(f"<root>{string1}</root>")
        string2_xml = etree.fromstring(f"<root>{string2}</root>")
        if not match_xml(string1_xml, string2_xml):
            return (None, None)

        string1_segmented = split_by_xml_tags(string1)
        string2_segmented = split_by_xml_tags(string2)
        string1_segmented = [seg for seg in string1_segmented if seg is not None]
        string2_segmented = [seg for seg in string2_segmented if seg is not None]

        if len(string1_segmented) == len(string2_segmented):
            return (string1_segmented, string2_segmented)
    except:
        pass

    return (None, None)

def prepare_xml_markup_pairs(hypotheses, references):
    hypothesis_segmented, references_segmented = [], []
    non_matching_indices = []
    current_index = 0
    for hyp, ref in zip(hypotheses, references):
        hs, rs = split_pairs_by_xml_tags(hyp, ref)
        if hs and rs:
            hypothesis_segmented.extend(hs)
            references_segmented.extend(rs)
            current_index += len(hs)
        else:
            non_matching_indices.append(current_index)
            current_index += 1
    return hypothesis_segmented, references_segmented, non_matching_indices

