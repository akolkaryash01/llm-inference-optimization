\# LLM Inference Optimization



Optimizing LLM inference on GPU (Colab) and CPU (local) using

quantization, batching, caching, and streaming.



\## Results — GPU (Google Colab T4)



| Method | Tokens/sec | VRAM |

|---|---|---|

| FP16 baseline | 14.5 | 5.57 GB |

| 4-bit NF4 | 7.3 | 1.84 GB |

| 4-bit batched x4 | 12.5 | 1.84 GB |



\*\*67% VRAM reduction with 4-bit quantization.\*\*



\## Results — CPU (Local Windows)



| Threads | Tokens/sec |

|---|---|

| 1 | x.x |

| 4 | x.x |

| 8 | x.x |



\## What I built

\- FP16 vs 4-bit NF4 quantization comparison

\- Batched inference for higher GPU utilization  

\- KV cache warmup to reduce first-request latency

\- Token streaming like ChatGPT

\- Prompt caching with MD5 hashing

\- REST API with FastAPI

\- CPU inference with llama-cpp-python

\- Quality evaluation — ROUGE and BERTScore

\- Live Gradio demo



\## Model

microsoft/phi-2 — 2.7B parameters



\## How to run on Colab

\[!\[Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR\_COLAB\_LINK)



\## How to run locally

pip install llama-cpp-python gradio fastapi

jupyter notebook



\## Tech Stack

Python · HuggingFace · bitsandbytes · llama-cpp-python

FastAPI · Gradio · ROUGE · BERTScore

