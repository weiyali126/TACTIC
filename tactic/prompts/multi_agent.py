STEP_BY_STEP_SYS_PROMPT = '''You are a native speaker of both {source_language} and {target_language}, with expertise in translating from {source_language} to {target_language}.'''

DRAFT_PROMPT = '''In this phase, your sole objective is to generate a draft translation that strictly adheres to the source text. Avoid adding any additional information not present in the source text, nor omit any content from it. Every detail must be fully preserved in the {target_language} translation. Below are several translation strategies. Please provide your best {translation_type} for the following source text.

Translation Strategies:
1. Literal Translation: Also known as direct translation or word-for-word translation, it prioritizes accurate meaning while preserving the original text's form and content in the target language.  
2.Sense-for-Sense Translation: Focuses on conveying the core meaning of the original text without adhering strictly to its linguistic form, ensuring greater fluency and naturalness in the target language.  
3.Free Translation: Emphasizes delivering the overall meaning and effect of the original text, allowing for significant rewriting or restructuring as needed.

## Pre-translation Research:
{pre_translation_result}

## Context Analysis:
{context_analysis}

## Extended Context:
{extended_context}

## Few-shot Examples:
{few_shot_examples}

## Source Text ({source_language}):
{source_text}

## Output format specification:
```json
{{
    "translation": "<Your translation>"
}}
```
The JSON object：json
'''

DRAFT_PROMPT_COMMON = '''In this phase, your task is to generate a draft translation that accurately conveys the meaning and nuances of the source text while respecting {target_language} grammar, vocabulary, and cultural sensitivities.  
Please translate the following source text into {target_language}.  

## Source Text ({source_language}):
{source_text}

## Output format specification:
```json
{{
    "translation": "<Your translation>"
}}
```
The JSON object：json
'''


REFINEMENT_SYS_PROMPT = '''You are a native speaker of both {source_language} and {target_language}, with expertise in translating from {source_language} to {target_language}. As an assistant dedicated to enhancing translation quality, you will be given a source sentence in {source_language}, a list of candidate translations in {target_language}, along with relevant research. Your task is to carefully analyze the provided information and refine the translation, ensuring it accurately and fully captures the original meaning of the source text. Your analysis should be in English.
'''

REFINEMENT_PROMPT = '''Refine the translation from {source_language} to {target_language}.

## Pre-translation Research:
{pre_translation_result}

## Context Analysis:
{context_analysis}

## Extended Context:
{extended_context}

## Few-shot Examples:
{few_shot_examples}

## Source text ({source_language}):
{source_text}

## Candidate translations ({target_language}): 
{candidate_translations}

## Output format specification:
```json
{{
    "analysis": "<Brief analysis of the candidate translations>",
    "translation": "<Your refined translation>"
}}
```
The JSON object：json
'''

REFINEMENT_PROMPT_COMMON = '''Refine the translation from {source_language} to {target_language}.

## Pre-translation Research:
{pre_translation_result}

## Context Analysis:
{context_analysis}

## Source text ({source_language}):
{source_text}

## Candidate translations ({target_language}): 
{candidate_translations}

## Output format specification:
```json
{{
    "analysis": "<Brief analysis of the candidate translations>",
    "translation": "<Your refined translation>"
}}
```
The JSON object：json
'''
