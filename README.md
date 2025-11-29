---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:363
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: \section{5. Correspondencia con constantes físicas}
  sentences:
  - \section{7. Simulación Python – Malla Icosaédrica y Autovalores}
  - \textbf{Interpretación:} Cada modo $n$ representa un posible estado de partícula,
    resonancia o nodo cognitivo.
  - rroc_V
- source_sentence: '\[

    S_{\rm eff} = \int d^4x \left[ \frac{R}{16 \pi G} + \lambda \log(r) + \mathcal{L}_{\rm
    matter} \right]

    \]'
  sentences:
  - 'Sea $H$ la matriz discreta sobre la red icosaédrica. Los autovalores $\{E_n\}$
    y vectores propios $\{\Psi_n\}$ cumplen:'
  - '\begin{lstlisting}[language=Python]

    import numpy as np

    import networkx as nx

    from scipy.linalg import eigh

    import matplotlib.pyplot as plt'
  - \section{6. Aplicaciones en resonancia y cognición}
- source_sentence: \section{6. Aplicaciones en resonancia y cognición}
  sentences:
  - '# Visualizar malla

    pos = nx.spring_layout(G, seed=42)

    nx.draw(G, pos, with_labels=True, node_color=''cyan'', edge_color=''gray'')

    plt.show()

    \end{lstlisting}'
  - '\begin{lstlisting}[language=Python]

    import numpy as np

    import networkx as nx

    from scipy.linalg import eigh

    import matplotlib.pyplot as plt'
  - \section{5. Correspondencia con constantes físicas}
- source_sentence: \textbf{Interpretación:} Cada modo $n$ representa un posible estado
    de partícula, resonancia o nodo cognitivo.
  sentences:
  - '\begin{lstlisting}[language=Python]

    import numpy as np

    import networkx as nx

    from scipy.linalg import eigh

    import matplotlib.pyplot as plt'
  - ln(phi)
  - '# Visualizar malla

    pos = nx.spring_layout(G, seed=42)

    nx.draw(G, pos, with_labels=True, node_color=''cyan'', edge_color=''gray'')

    plt.show()

    \end{lstlisting}'
- source_sentence: "donde:\n\\begin{itemize}\n    \\item $\\psi_i$ son espinores icosaédricos.\n\
    \    \\item $r_{ij}$ es la distancia entre nodos $i$ y $j$.\n    \\item $A,B,C$\
    \ son acoplamientos gauge discretos.\n\\end{itemize}"
  sentences:
  - "\\begin{abstract}\nEsta versión extendida del \\textbf{Resonance of Reality Framework\
    \ (RRF)} presenta:\n\\begin{itemize}\n    \\item Hamiltoniano discreto icosaédrico\
    \ con modos normales.\n    \\item Corrección logarítmica gravitatoria y acoplamientos\
    \ gauge explícitos.\n    \\item Correspondencia con constantes físicas fundamentales.\n\
    \    \\item Ejemplo de simulación Python que visualiza la malla icosaédrica y\
    \ autovalores.\n\\end{itemize}\n\\end{abstract}"
  - "\\begin{itemize}\n    \\item Cada nodo $\\psi_i$ como \\textbf{átomo de experiencia}.\n\
    \    \\item Patrones icosaédricos y $\\phi$ guían frecuencia de resonancia.\n\
    \    \\item Protocolos musicales y visuales para plasticidad neuronal.\n\\end{itemize}"
  - "\\begin{itemize}\n    \\item Cada nodo $\\psi_i$ como \\textbf{átomo de experiencia}.\n\
    \    \\item Patrones icosaédricos y $\\phi$ guían frecuencia de resonancia.\n\
    \    \\item Protocolos musicales y visuales para plasticidad neuronal.\n\\end{itemize}"
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Deployment & quick start

### Run a local embedding API

1. Install dependencies (Python 3.10+ recommended):
   ```bash
   pip install -r requirements.txt
   ```
2. Start the FastAPI server (loads the model from this directory by default):
   ```bash
   uvicorn deploy:app --host 0.0.0.0 --port 8000
   ```
   Set `MODEL_PATH=/path/to/model` to point at a different checkpoint.
3. Call the endpoints:
   - Get metadata: `curl http://localhost:8000/`
   - Embed sentences:
     ```bash
     curl -X POST http://localhost:8000/embed \
       -H "Content-Type: application/json" \
       -d '{"sentences": ["hola mundo", "resonance of reality"], "normalize": true}'
     ```
   - Similarity against a list of targets:
     ```bash
     curl -X POST http://localhost:8000/similarity \
       -H "Content-Type: application/json" \
       -d '{"source": "icosahedral network", "targets": ["graph structure", "text embedding"], "normalize": true}'
     ```

### Use in Python

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer(".")  # or a custom path
queries = ["icosahedral lattice", "autovalores de la red"]
embeddings = model.encode(queries, convert_to_tensor=True, normalize_embeddings=True)
score = util.cos_sim(embeddings[0], embeddings[1])
print(score)
```

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'donde:\n\\begin{itemize}\n    \\item $\\psi_i$ son espinores icosaédricos.\n    \\item $r_{ij}$ es la distancia entre nodos $i$ y $j$.\n    \\item $A,B,C$ son acoplamientos gauge discretos.\n\\end{itemize}',
    '\\begin{itemize}\n    \\item Cada nodo $\\psi_i$ como \\textbf{átomo de experiencia}.\n    \\item Patrones icosaédricos y $\\phi$ guían frecuencia de resonancia.\n    \\item Protocolos musicales y visuales para plasticidad neuronal.\n\\end{itemize}',
    '\\begin{abstract}\nEsta versión extendida del \\textbf{Resonance of Reality Framework (RRF)} presenta:\n\\begin{itemize}\n    \\item Hamiltoniano discreto icosaédrico con modos normales.\n    \\item Corrección logarítmica gravitatoria y acoplamientos gauge explícitos.\n    \\item Correspondencia con constantes físicas fundamentales.\n    \\item Ejemplo de simulación Python que visualiza la malla icosaédrica y autovalores.\n\\end{itemize}\n\\end{abstract}',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.7950, 0.7297],
#         [0.7950, 1.0000, 0.7343],
#         [0.7297, 0.7343, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 363 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 363 samples:
  |         | sentence_0                                                                          | sentence_1                                                                         | label                                                         |
  |:--------|:------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                              | string                                                                             | float                                                         |
  | details | <ul><li>min: 16 tokens</li><li>mean: 65.29 tokens</li><li>max: 127 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 38.02 tokens</li><li>max: 127 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.5</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                               | sentence_1                                                                                                                                                                                                                                      | label                           |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------|
  | <code>Sea $H$ la matriz discreta sobre la red icosaédrica. Los autovalores $\{E_n\}$ y vectores propios $\{\Psi_n\}$ cumplen:</code>                                                     | <code># Crear grafo icosaédrico<br>G = nx.icosahedral_graph()<br>n = G.number_of_nodes()</code>                                                                                                                                                 | <code>0.4647801650468898</code> |
  | <code>\[<br>i \hbar \frac{\partial \Psi}{\partial t} = H \Psi<br>\]</code>                                                                                                               | <code>\begin{align*}<br>\alpha_{\rm fine} &\approx f(E_n, \text{geometría icosaédrica}) \\<br>m_\nu &\approx g(\text{acoplamientos SU(2)/SU(3) discretos}) \\<br>\Lambda &\approx h(\text{energía de vacío logarítmica})<br>\end{align*}</code> | <code>0.4930957329947213</code> |
  | <code>\title{Resonance of Reality Framework (RRF) Extendido\\<br>Hamiltoniano Icosaédrico, Gravedad Logarítmica y Simulación}<br>\author{Antony Padilla Morales}<br>\date{\today}</code> | <code>donde:<br>\begin{itemize}<br>    \item $\psi_i$ son espinores icosaédricos.<br>    \item $r_{ij}$ es la distancia entre nodos $i$ y $j$.<br>    \item $A,B,C$ son acoplamientos gauge discretos.<br>\end{itemize}</code>                  | <code>0.5762137786115148</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Framework Versions
- Python: 3.12.12
- Sentence Transformers: 5.1.1
- Transformers: 4.57.0
- PyTorch: 2.8.0+cu126
- Accelerate: 1.10.1
- Datasets: 4.0.0
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}

@misc{antony_padilla_morales_2025,
	author       = { Antony Padilla Morales },
	title        = { RRFSAVANTMADE (Revision 13af35f) },
	year         = 2025,
	url          = { https://huggingface.co/antonypamo/RRFSAVANTMADE },
	doi          = { 10.57967/hf/7034 },
	publisher    = { Hugging Face }
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->