# How the benchmark runs

Quality scores (nDCG, Recall, STS Spearman) never go through Qdrant. Qdrant is optional and only used for serving-path numbers when you pass `--profile-ops`.

How to peek at the actual texts: [inspect-datasets.md](inspect-datasets.md).  
How to add your own folder: [custom-datasets.md](custom-datasets.md).

## Components

```mermaid
flowchart LR
  CLI["embbench CLI"] --> Spec["JobSpec"]
  API["FastAPI stub"] --> Spec
  Spec --> Runner["run_job"]
  Runner --> Encoder["one encoder on GPU"]
  Runner --> Tasks
  Tasks --> MTEB["MtebSource<br/>configs/tasks.yaml"]
  Tasks --> Local["LocalSource<br/>data/sts and data/retrieval"]
  Encoder --> Eval["mteb.evaluate"]
  MTEB --> Eval
  Local --> Eval
  Eval --> Exact["exact cosine in RAM"]
  Exact --> Scores["nDCG / Recall / Spearman"]
  Scores --> Out["results/job/result.json"]
  Runner -.->|"--profile-ops only"| Qdrant["Qdrant ANN"]
  Qdrant -.-> Ops["p95 / index size / ANN delta"]
  Ops -.-> Out
```

`run_job(JobSpec) -> JobResult` is the only execution path. The CLI and the API only build a spec.

## Request sequence: `embbench run`

`--model all` spawns **one subprocess per model** and waits for exit before the next, so two encoders never share the GPU.

```mermaid
sequenceDiagram
  autonumber
  actor You
  participant CLI as embbench CLI
  participant Orch as orchestrator subprocess
  participant Worker as worker process
  participant Runner as run_job
  participant Enc as encoder GPU
  participant Src as MtebSource / LocalSource
  participant MTEB as mteb.evaluate
  participant Disk as results/

  You->>CLI: uv run embbench run --model all
  CLI->>Orch: spawn, EMBBENCH_WORKER=1
  loop one model at a time
    Orch->>Worker: python -m embbench.cli run --model bce-...
    Worker->>Runner: JobSpec languages, task types, flags
    Runner->>Disk: manifest.json
    Runner->>Src: list_tasks then load_mteb_task
    Src-->>Runner: FiQA, Belebele, local folders, ...
    Runner->>Enc: load_encoder once
    loop each task sequentially
      Enc->>MTEB: encode corpus and queries
      MTEB->>MTEB: cosine top-k in memory
      MTEB->>MTEB: pytrec_eval nDCG Recall or Spearman
      MTEB-->>Disk: predictions/*.json
      Runner-->>Disk: append TaskScore to result.json
      Runner->>Enc: empty CUDA cache
    end
    opt --profile-ops
      Runner->>Runner: ExactMemoryBackend latency
      Runner->>Runner: Qdrant upsert and search if up
    end
    Worker-->>Orch: exit 0 or fail-soft
  end
  Orch->>Disk: results/report.md
```

A single `--model bce-embedding-base_v1` skips the orchestrator and calls `run_job` in the same process.

## Which datasets get loaded

```mermaid
flowchart TD
  Spec["JobSpec"] --> Lang["--languages eng,cmn,zsm"]
  Spec --> Types["--task-types Retrieval,STS"]
  Lang --> Collect["_collect_tasks"]
  Types --> Collect
  Collect --> Mteb{"include_mteb?"}
  Mteb -->|yes| YAML["configs/tasks.yaml"]
  YAML --> Heavy{"heavy and not --include-heavy?"}
  Heavy -->|skip| Drop["T2Retrieval, MKQARetrieval, ..."]
  Heavy -->|keep| Pub["public MTEB tasks"]
  Collect --> Loc{"include_local?"}
  Loc -->|yes| Folders["data/retrieval/* and data/sts/*"]
  Folders --> Alias["language aliases to eng / cmn / zsm"]
  Alias --> LocalTasks["local tasks"]
  Collect --> Names{"--task-names set?"}
  Pub --> Names
  LocalTasks --> Names
  Names -->|filter by name| Final["task list"]
  Names -->|no filter| Final
```

Malay public tasks today: BelebeleRetrieval (`zsm`), WebFAQRetrieval (`msa`). MKQARetrieval is heavy. Malay STS is empty until you drop `data/sts/<name>/`.

## Retrieval scoring vs Qdrant

```mermaid
sequenceDiagram
  autonumber
  participant Task as retrieval task
  participant Enc as encoder
  participant Mem as RAM cosine
  participant Eval as pytrec_eval
  participant Q as Qdrant

  Note over Task,Eval: Scoring path. Always used. No Qdrant.
  Task->>Task: load corpus, queries, qrels
  Task->>Enc: encode all documents
  Task->>Enc: encode all queries
  Enc->>Mem: score = docs @ query  (normalized)
  Mem->>Mem: top-k doc ids per query
  Mem->>Eval: run vs qrels
  Eval-->>Task: nDCG@10 nDCG@30 Recall@10 Recall@30

  Note over Q: Serving path. Only with --profile-ops.
  Enc->>Q: upsert a sample of vectors
  Enc->>Q: search a few queries
  Q-->>Task: p95 ms, index size, ANN vs exact recall delta
```

STS is the same encoder with no search: embed sentence1 and sentence2, cosine, Spearman vs the human `score`.

## Where files land

```mermaid
flowchart LR
  Hub["HuggingFace Hub"] --> Cache["/mnt/c/ml-cache/huggingface"]
  Cache --> Enc["model weights"]
  Cache --> DS["MTEB datasets"]
  Runner["run_job"] --> Job["results/RUNID-model/"]
  Job --> RJ["result.json"]
  Job --> Pred["predictions/"]
  MTEB["mteb.ResultCache"] --> MC["results/mteb-cache/"]
  Report["embbench report"] --> MD["results/report.md"]
```

Weights and public datasets stay on the Windows drive (`HF_HOME` must start with `/mnt/c`). Job outputs stay on WSL ext4 under `results/`.
