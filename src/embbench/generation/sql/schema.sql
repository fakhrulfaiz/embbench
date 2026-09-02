-- Dataset-store schema (documents, chunks, labels, export ledger).
-- Generation reads chunks and writes sts_pairs, retrieval_questions,
-- and dataset_exports. CREATE IF NOT EXISTS: safe on a live DB.

CREATE TABLE IF NOT EXISTS documents (
    doc_id uuid PRIMARY KEY,
    source_url text,
    filename text,
    extractor_task_id text,
    extracted_text text,
    pages_processed int,
    language text,
    doc_type text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id uuid PRIMARY KEY,
    doc_id uuid NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    chunker text,
    content text NOT NULL,
    language text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS chunks_profile_lang_idx ON chunks (chunker, language);

CREATE TABLE IF NOT EXISTS retrieval_questions (
    question_id uuid PRIMARY KEY,
    gold_chunk_id uuid NOT NULL REFERENCES chunks (chunk_id) ON DELETE CASCADE,
    question_text text NOT NULL,
    language text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS retrieval_questions_gold_idx
    ON retrieval_questions (gold_chunk_id);
CREATE INDEX IF NOT EXISTS retrieval_questions_lang_idx
    ON retrieval_questions (language);

-- pair_kind: paraphrase (score 5), generated (3–4), chunk_chunk (0–2).
CREATE TABLE IF NOT EXISTS sts_pairs (
    pair_id uuid PRIMARY KEY,
    sentence1 text NOT NULL,
    sentence2 text NOT NULL,
    score float8 NOT NULL,
    source_chunk_id uuid REFERENCES chunks (chunk_id) ON DELETE SET NULL,
    pair_kind text NOT NULL,
    language text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS sts_pairs_lang_idx ON sts_pairs (language);
CREATE INDEX IF NOT EXISTS sts_pairs_score_idx ON sts_pairs (score);

CREATE TABLE IF NOT EXISTS dataset_exports (
    export_id uuid PRIMARY KEY,
    name text NOT NULL,
    task_type text NOT NULL,
    language text NOT NULL,
    revision text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
