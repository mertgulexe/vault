# Referene Links & Notes
* Bootcamp: `Train, Fine-tune and Deploy LLMs - Bootcamp`
* Instructor: [Damien Benveniste](https://www.linkedin.com/in/damienbenveniste/)
* Curriculum: [Link](https://learn.theaiedge.io/courses/the-large-language-bootcamp/lectures/55239139)

## Week 1: The Transformer Architecture
* [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
* GPT-3 architecture: [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165)
* Strided sparse attention mechanism: [Generating Long Sequences with Sparse Transformers](https://arxiv.org/pdf/1904.10509)
* [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/pdf/2307.09288)
* Constant space complexity & Llama and Mistral use a memory-efficient attention mechanism: [SELF-ATTENTION DOES NOT NEED $O(n^2)$ MEMORY](https://arxiv.org/pdf/2112.05682)
* [Mistral 7B](https://arxiv.org/pdf/2310.06825)
* Gemini and Mistral 7B uses the Multi-query attention mechanism: [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/pdf/1911.02150)
* Terraformer suggests that the last linear layer is unnecessary: [Sparse is Enough in Scaling Transformers](https://arxiv.org/pdf/2111.12763)
* [Mistral implementation](https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/model.py)
* RoPE (Rotary Position Embedding): [ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING](https://arxiv.org/pdf/2104.09864)
* RMSNorm layer (Root Mean Square Layer Normalization): [Root Mean Square Layer Normalization](https://arxiv.org/pdf/1910.07467)

## Week 2: Training LLMs to Follow Instructions

|Dataset|Quantity (tokens)|Weight in training mix|URL|
|-|-|:-:|-|
|Common Crawl (filtered)|410 billion|60%|[https://commoncrawl.org/overview](https://commoncrawl.org/overview)|
|WebText2|19 billion|22%|[https://openwebtext2.readthedocs.io/en/latest/](https://openwebtext2.readthedocs.io/en/latest/)|
|Books1|12 billion|8%|[https://github.com/soskek/bookcorpus](https://github.com/soskek/bookcorpus)|
|Books2|55 billion|8%|[https://github.com/soskek/bookcorpus](https://github.com/soskek/bookcorpus)|
|Wikipedia|3 billion|3%|[https://huggingface.co/datasets/wikipedia](https://huggingface.co/datasets/wikipedia)|

<p style="text-align: center; font-style: italic;">Table-1: Data To train GPT-3</p>

* Alpaca model and dataset: [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)
* InstructGPT paper: [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155)
* [PIQA: Reasoning about Physical Commonsense in Natural Language](https://arxiv.org/pdf/1911.11641v1)
* [PapersWithCode - Question Answering on PIQA](https://paperswithcode.com/sota/question-answering-on-piqa)
* [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
* [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
* [GPT-2 Model card](https://huggingface.co/openai-community/gpt2)
* [The PIQA dataset on HuggingFace](https://huggingface.co/datasets/ybisk/piqa)
* DPO: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290)
* [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/pdf/2403.07691)
    * _The loss function associated with ORPO is simply using the logarithm of the [odds ratio](https://en.wikipedia.org/wiki/Odds_ratio)._

## Week 3: How to Scale Model Training
* <a href="https://arxiv.org/pdf/1704.04760" ref="noreferrer" target="_blank">In-Datacenter Performance Analysis of a Tensor Processing Unit<sup>TM</sup></a>
* [Efficient Training on Multiple GPUs](https://huggingface.co/docs/transformers/v4.40.2/en/perf_train_gpu_many)
* Zero Redundancy Optimizer: [ZeRO & DeepSpeed: New system optimizations enable training models with over 100 billion parameters](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)
* [Benchmarking TPU, GPU, and CPU Platforms for Deep Learning](https://arxiv.org/pdf/1907.10701)
* [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/pdf/2307.08691)
* [Distributed Training of Deep Learning Models: A Taxonomic Perspective](https://arxiv.org/pdf/2007.03970)
* [Emotion dataset](https://huggingface.co/datasets/dair-ai/emotion)
* [DeepSpeed package](https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed)
* Adam optimizer: [Adam: A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980)

## Week 4: How to Fine-Tune LLMs
* [An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning](https://arxiv.org/pdf/2308.08747)
* [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685)
* [PEFT (Parameter-Efficient Fine-Tuning)](https://huggingface.co/docs/peft/en/index)
* [LONGLORA: EFFICIENT FINE-TUNING OF LONGCONTEXT LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2309.12307v2)
    * [Github repository](https://github.com/dvlab-research/LongLoRA)
*  [Llama 3-8B model card](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
* [Modeling Llama](https://github.com/huggingface/transformers/blob/v4.34-release/src/transformers/models/llama/modeling_llama.py)
* [LongAlpaca-12k dataset](https://huggingface.co/datasets/Yukang/LongAlpaca-12k)
* AdamW: [DECOUPLED WEIGHT DECAY REGULARIZATION](https://arxiv.org/pdf/1711.05101)

## Week 5: How to Deploy LLMs
* Using WebGPU
    * https://github.com/mlc-ai/web-llm
    * https://huggingface.co/Xenova
* [Seamlessly Deploying a Swarm of LoRA Adapters with NVIDIA NIM](https://developer.nvidia.com/blog/seamlessly-deploying-a-swarm-of-lora-adapters-with-nvidia-nim)
* [Efficiently Serving LLMs](https://www.deeplearning.ai/short-courses/efficiently-serving-llms/)
* [vLLM package](https://github.com/vllm-project/vllm) and [vLLM docs](https://docs.vllm.ai/en/latest/index.html)
* [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/pdf/2309.06180)
* Some deployment options:
    * [Hugging Face: Inference Endpoints](https://ui.endpoints.huggingface.co/)
    * [AWS Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-import-model.html)
    * [Replicate](https://replicate.com/)
* [Phi-3-mini-4k-instruct dataset](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

## Week 6: Building the Application Layer
* Hypothetical Document Embeddings (HyDE): [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/pdf/2212.10496)
* 🦜️🔗 LangChain: [How to use the MultiQueryRetriever](https://python.langchain.com/docs/how_to/MultiQueryRetriever/)
* SQLAlchemy: [Object Relational Tutorial](https://docs.sqlalchemy.org/en/14/orm/tutorial.html)
* Pinecone Docs: [Understanding metadata](https://docs.pinecone.io/guides/data/understanding-metadata)