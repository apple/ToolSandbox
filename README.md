# Reproduction Steps for ToolSandbox

Note that the original README is at `README.original.md`.

## Setup

- Make a new Conda environment with Python 3.11, i.e.,
```
conda create -n tool_sandbox python=3.11
conda activate tool_sandbox
```
- Install `pip` via `conda install pip`.
- Change directory to the `tool_sandbox` directory.
- Run `pip install ".[dev]"`.

## Benchmarking

- To benchmark Athene V2 Agent, run `RAPID_API_KEY=<rapid_api_key> ATHENE_V2_AGENT_ENDPOINT=<endpoint> ATHENE_V2_AGENT_API_KEY=<api_key> tool_sandbox --user GPT_4_o_2024_05_13 --agent AtheneV2Agent`
- To benchmark OpenAI GPT-4o, run `RAPID_API_KEY=<rapid_api_key> OPENAI_API_KEY=<openai_api_key> tool_sandbox --user GPT_4_o_2024_05_13 --agent GPT_4_o_2024_08_06`

The `rapid_api_key` is any valid RapidAPI key. The `endpoint` is the endpoint provided to you by Nexusflow. It should be of the form `<url>:<port>/v1`, e.g., "http://dgx1.nexusflow.ai:10249/v1". The `api_key` is the key given to you by Nexusflow. The `openai_api_key` is any valid OpenAI API key.

## Generating results 

Note that each benchmark run generates a new result cache in `data`. To benchmark two runs against each other, run `python tabulate_results.py data/<run_1>/result_summary.json data/<run_2>/result_summary.json`. The outputs label the first run's agent as `Athene-Agent` and the second run as `GPT4o`, but this is cosmetic (it just represents the usual case for our evaluation).