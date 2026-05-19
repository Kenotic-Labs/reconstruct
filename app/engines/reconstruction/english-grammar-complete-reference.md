# Information Retrieval and Reconstruction in AI Memory Systems: A Comprehensive Technical Reference

**TL;DR**
- For a deterministic reconstruction engine over conversational memory (the user's stated goal), the right backbone is **structured extraction → entity-keyed SQL/FTS5 store → hybrid BM25+dense retrieval with Reciprocal Rank Fusion → cross-encoder reranker → verification step that confirms the candidate answers the question's predicate-argument structure** — this is essentially the Lyzr Cognis architecture (arXiv:2604.19771), which currently posts state-of-the-art numbers on LOCOMO and LongMemEval and is the closest published instantiation of the design pattern the user is building.
- The LOCOMO benchmark exposes that *most popular memory systems silently exclude the "adversarial" (unanswerable) category* because their pipelines lack a refusal mechanism — a deterministic reconstruction engine must treat **abstention as a first-class output**, not a confidence threshold, because token-overlap scoring and cosine-similarity thresholds provably fail on questions designed to look semantically similar to stored content but have no factual answer.
- The seven "banned/dangerous" methods you listed (threshold scoring, token-overlap verification, self-verification loops, test contamination, cosine-similarity overreliance, greedy retrieval, retrieval without verification) are not stylistic preferences — they map to documented failure modes in published benchmarks; the recommended replacements are *structural verification* (predicate-argument matching), *external verification* (SQL fact lookup), and *typed retrieval* with schema constraints.

---

## Key Findings

1. **Hybrid lexical + dense + cross-encoder is the empirically dominant retrieval stack for conversational memory in 2025–2026.** Cognis (BM25/FTS5 + Matryoshka vector + RRF at 70/30 + BGE-2 cross-encoder + temporal boost) and MemMachine (0.8487 overall on LoCoMo, 0.9169 with gpt-4.1-mini per their March 2026 blog) both beat the older Mem0 (vector+LLM extract) and Zep (Graphiti temporal KG) on LongMemEval and LOCOMO. The unifying pattern is *cheap broad recall → expensive precise rerank → explicit verification*, not any single neural retrieval model.
2. **Deterministic retrieval without any LLM call is viable up to roughly 80% of LOCOMO single-hop quality** if you commit to (a) structured extraction at write-time into entity-keyed rows, (b) BM25 over FTS5 plus optional sparse-dense fusion, (c) predicate-argument template matching at read-time, and (d) SQL-based fact verification. The LLM is only required for the *extraction* step and for natural-language surface form; the retrieval-and-verification path can be fully deterministic.
3. **The LOCOMO "adversarial" category is the canary for retrieval-system honesty.** Maharana et al. (2024) report LLM F1 ≈ 2.1 vs human 89.4 on adversarial. The Mem0 paper explicitly excludes this category, calling it "designed to test systems' ability to recognize unanswerable questions" — and that exclusion led to the Zep–Mem0 methodology dispute (Mem0 CTO Deshraj on GitHub: Zep's "mean accuracy is 58.44% ± 0.20" with adversarial removed, vs Zep's claim of 75.14% ± 0.17). The ATANT v1.1 audit (arXiv:2604.10981) found that for 444 of 446 category-5 questions the gold answer is empty, making them "unscorable by construction" by current evaluation scripts — so most published "adversarial" numbers measure scoring-script artifacts, not refusal.
4. **Speaker attribution is the silent failure mode of multi-speaker memory.** EverMemBench (arXiv:2602.01313) reports multi-hop accuracy collapsing to 26% under multi-party attribution even with oracle evidence. The Penfield Labs audit of LOCOMO found 24 questions where the benchmark itself attributes statements to the wrong speaker, indicating that even ground-truth annotation is fragile here.
5. **The "banned" methods fail in predictable, named ways.** Threshold scoring is fragile because BM25 and cosine scores are *not calibrated across corpora*; token-overlap verification fails because LOCOMO adversarial questions are specifically constructed to have "high surface-level semantic similarity with part of the conversation" (Bini et al., 4 Dec 2025); self-verification loops fail because the verifier and candidate share the same embedding space and confirm each other's hallucinations; greedy retrieval fails because BM25's top-1 is dominated by rare terms (the "orangutan effect"), not semantic centrality.

---

## Details

### 1. Basic Methods

**TF–IDF.** Term Frequency × Inverse Document Frequency. TF is how often a term appears in a document; IDF is log(N/df). Score is the sum over query terms of TF·IDF. *Good at:* discriminating rare terms; trivial to compute; transparent. *Fails at:* document length bias (longer docs get higher scores), unbounded IDF for very rare terms, no vocabulary mismatch handling, no semantic similarity. *When to use:* tiny corpora, debugging baselines, or when you need a sanity check that semantic search is actually beating lexical.

**BM25 (Robertson, Okapi).** Introduces (a) term-frequency saturation with parameter k₁ (typically 1.2–1.5), so 100 occurrences are not 10× more relevant than 10; (b) document-length normalization with parameter b (typically 0.75); (c) a smoothed IDF. SQLite's FTS5 hardcodes k₁=1.2, b=0.75, and returns *negative* BM25 (smaller = better, so `ORDER BY bm25(table)` ascending gets best matches). *Good at:* strong, well-understood baseline that frequently matches or exceeds dense retrievers on out-of-distribution data; near-zero infrastructure cost. *Fails at:* vocabulary mismatch (synonyms, paraphrase), pronouns, ellipsis. *When to use:* always have it as one channel of a hybrid retriever; it is the BEIR benchmark's hardest-to-beat baseline.

**Boolean retrieval.** Pure set-theoretic AND/OR/NOT over an inverted index. FTS5 supports this directly with `MATCH 'sqlite AND searching'`, `NEAR(term1 term2, 5)`, prefix `data*`, and `NOT`. *Good at:* precision-critical workloads (legal, compliance) where false positives are catastrophic. *Fails at:* recall, ranking, fuzzy match. *When to use:* as a *filter* before ranked retrieval, not as a ranker.

**Inverted index.** The data structure underneath both Boolean and BM25 retrieval: a hash from term → posting list of (doc_id, term_positions, term_frequency). FTS5 stores this in a log-structured merge tree across shadow tables (`%_data`, `%_content`, `%_docsize`, `%_idx`, `%_config`). Updates write a new segment; periodic merges keep search fast. *Good at:* sub-linear-in-corpus-size lookup. *Fails at:* dense floating-point similarity, dynamic re-weighting at query time.

**FTS5 in SQLite.** `CREATE VIRTUAL TABLE x USING fts5(col1, col2, tokenize='porter');` with the `content='base_table'` option creates an external-content table that re-uses the canonical row store. Supports `unicode61` (default, Unicode 6.1), `ascii`, and `porter` (stemming) tokenizers; supports `MATCH`, `bm25()`, `snippet()`, `highlight()`, and `NEAR()`. *Critical implementation note:* triggers on the base table (`AFTER INSERT/UPDATE/DELETE`) are required to keep the FTS5 shadow tables in sync, or you must call `INSERT INTO x(x) VALUES('rebuild')` after batch loads.

**Simple SQL lookup.** Exact equality on indexed columns: `SELECT body FROM messages WHERE speaker = ? AND date = ?`. Use a B-tree index on the lookup keys. *Good at:* deterministic precision on structured fields (entity, date, session); zero ambiguity. *Fails at:* anything requiring fuzzy match. *When to use:* always as the *first* retrieval channel in a deterministic engine — if the question has an unambiguous entity+date+predicate, you should answer from SQL before invoking any other retrieval.

**Pattern matching and string similarity.** `LIKE '%term%'` (slow: full table scan, no index), `GLOB`, regular expressions, and edit-distance functions (Levenshtein, Jaro-Winkler). For approximate matching at scale, use n-gram indexes or trigram (PostgreSQL's `pg_trgm`). *Good at:* prefix/suffix queries, typo tolerance, code search. *Fails at:* semantic match, ranking by relevance. *When to use:* as a fallback channel when BM25 misses due to morphological variation that the tokenizer didn't normalize.

### 2. Intermediate Methods

**Dense retrieval / bi-encoder.** A single model (typically a sentence transformer like `all-MiniLM-L6-v2`, `bge-large`, or `text-embedding-3-large`) maps each query and each document to a single vector. Retrieval is a maximum inner product (MIPS) or cosine search. *Mechanics:* indexed offline; at query time, encode the query (single forward pass) and run ANN search. *Good at:* paraphrase, synonyms, zero-shot generalization across domains for strong models. *Fails at:* exact entity/number matching — Sciavolino, Zhong, Lee & Chen ("Simple Entity-centric Questions Challenge Dense Retrievers," EMNLP 2021, arXiv:2109.08535) show that even a DPR trained on the 65M-pair PAQ dataset "still performs far worse than BM25" on their EntityQuestions diagnostic, and BM25 consistently outperforms all DPR variants on top-20 retrieval accuracy for entity-centric questions. Also fails on out-of-distribution domains, and on explainability. *When to use:* always paired with BM25, never alone, for conversational memory.

**Sparse–dense hybrid.** Run BM25 and dense retrieval independently, then fuse. Empirically, hybrid beats either alone on BEIR, MS MARCO, and LOCOMO. Cognis uses 70% vector / 30% BM25 with RRF. The two failure modes are anti-correlated: BM25 misses paraphrase, dense misses rare entities and numbers, so the union covers both.

**Cross-encoder reranking.** After retrieving the top 50–200 candidates with the cheap first stage, run a cross-encoder (e.g., `ms-marco-MiniLM-L-12-v2`, `bge-reranker-large`, `BGE-2` cross-encoder used by Cognis) that takes (query, document) as a *concatenated* input through BERT and produces a single relevance score. *Good at:* dramatically improving precision@1 and @5; corrects the systematic biases of bi-encoders. *Fails at:* latency (you cannot precompute, every query × document needs a full forward pass) and recall (it can only rerank what the first stage retrieved). *When to use:* always, when you can afford ~50–200ms per query.

**Query expansion and reformulation.** Three flavors: (a) lexical expansion via WordNet, RM3 pseudo-relevance feedback, or learned expansion (Doc2Query, generating likely questions for each document at index time); (b) LLM rewriting (Query2Doc, "Rewrite this query for retrieval"); (c) HyDE (see below). *Good at:* short, ambiguous queries. *Fails at:* introducing noise that drags relevant results down; can hallucinate non-existent terms.

**Reciprocal Rank Fusion (RRF).** Introduced by Cormack, Clarke & Büttcher in their 2009 SIGIR paper "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods." For each result *d* in each ranked list *L*, score = Σ 1/(k + rank_L(d)), with k typically 60. *Good at:* combining incommensurable score scales (BM25 scores ≠ cosine scores) without normalization; zero-shot, zero-training. *Fails at:* if one retriever is dramatically better than the other, RRF dilutes it; weighted RRF helps but reintroduces a tuning parameter.

**ANN indexes — FAISS, HNSW, Annoy, ScaNN.** FAISS (Meta) supports Flat (exact, brute force), IVF (inverted file with k-means clustering), PQ (product quantization for memory compression), and HNSW (graph-based). HNSW (Malkov & Yashunin 2018) builds a multi-layer skip-list-of-proximity-graphs and provides approximately O(log N) approximate search with excellent dynamic insert/delete behavior. Annoy (Spotify) uses random projection forests; lower memory, mostly static. ScaNN (Google) uses anisotropic vector quantization. On ANN-Benchmarks (Aumüller, Bernhardsson & Faithfull, ann-benchmarks.com), hnswlib and hnsw(faiss) consistently appear in the top cluster on sift-128-euclidean and glove-100-angular plots, achieving the highest queries-per-second at recall@10 ≥ 0.95.

**Knowledge graph traversal.** Store facts as (subject, predicate, object) triples; answer multi-hop questions by graph walk. Cypher (Neo4j) or SPARQL (RDF). For conversational memory, Zep/Graphiti stores nodes as entities and edges as relationships with explicit (t_valid, t_invalid) intervals — its *bitemporal* model tracks both *event time* and *ingestion time*. *Good at:* multi-hop ("Who is Meghan Markle's husband's grandmother?"), temporal validity, contradictions and updates. *Fails at:* coverage (anything not extracted into triples is invisible), free-form text retrieval.

**Predicted query matching.** At write-time, an LLM generates likely questions about each chunk; at query-time, you match the user's question against the *pre-generated* questions, not the document body. This collapses the semantic gap because both sides are now questions. Doc2Query (Nogueira et al. 2019) was the first; the modern descendant is the "Answerable Question Generation" stage in question-centric RAG. *Good at:* short ambiguous user queries against long documents. *Fails at:* the LLM may not anticipate the question actually asked; doubles index storage.

### 3. Advanced Methods

**ColBERT and late interaction (Khattab & Zaharia, SIGIR 2020).** Instead of pooling to one vector, ColBERT keeps *token-level* embeddings for both query and document. Scoring is `MaxSim`: for each query token, take its maximum cosine with any document token, sum across query tokens. *Good at:* fine-grained matching; outperforms single-vector dense retrievers on out-of-domain (BEIR). *Fails at:* storage (multi-vector indexes are 10× the size; ColBERTv2 uses residual compression and centroid quantization to mitigate); slower retrieval than single-vector ANN.

**SPLADE (Formal et al., SIGIR 2021).** Uses a BERT-style MLM head to project each token onto the full vocabulary and produces a *sparse* vector with explicit term-expansion weights. Stored in an inverted index just like BM25, so it inherits BM25's tooling. *Good at:* term expansion fixes vocabulary mismatch while staying interpretable and using existing infrastructure. *Fails at:* slower than BM25 at query time because vectors are denser than typical sparse BM25 postings. Variants include SPLADE-doc, SPLADE-max, DistilSPLADE-max, and Echo-Mistral-SPLADE (decoder-only LLM backbone, current BEIR SOTA among LSR).

**HyDE — Hypothetical Document Embeddings (Gao et al., 2022, "Precise Zero-Shot Dense Retrieval without Relevance Labels").** Step 1: prompt an LLM, "Write a passage that answers: {query}." Step 2: embed that hypothetical passage. Step 3: retrieve documents nearest to the *hypothetical*, not to the query. *Good at:* zero-shot dense retrieval where you have no labeled (query, doc) pairs; bridges short-query-to-long-document semantic gap. *Fails at:* if the LLM hallucinates a confident-but-wrong hypothetical, it anchors retrieval to nothing; doubles latency and cost.

**Self-RAG (Asai et al., ICLR 2024, arXiv:2310.11511).** Trains an LM to emit four kinds of "reflection tokens": Retrieve (yes/no), IsRel (is the retrieved passage relevant?), IsSup (is the generated text supported?), IsUse (is the answer useful?). Adaptive: skips retrieval when parametric knowledge suffices. SeaKR (arXiv:2406.19215) and Probing-RAG (Baek et al. 2024, which skips retrieval in 57.5% of cases while exceeding adaptive RAG baselines by 6–8 points) probe internal hidden states instead of training reflection tokens.

**Multi-vector retrieval.** Beyond ColBERT, represent each document as: (a) the full text vector + a title vector + per-paragraph vectors (LangChain "parent document retriever"); (b) a summary vector + a raw-chunk vector; (c) one vector per anticipated question (Doc2Query at retrieval time). *Good at:* matching different granularities of queries.

**RAG with chain-of-thought.** Generate intermediate reasoning that itself triggers further retrieval (IRCoT — Trivedi et al. 2023). Each reasoning step issues a sub-query whose retrieved evidence conditions the next step. *Good at:* multi-hop QA. *Fails at:* error compounding — Reasoning Tree Guided RAG (arXiv:2601.11255) names the two failure modes "inaccurate query decomposition" and "error propagation."

**Decomposed retrieval.** Plan-and-execute: an LLM (or a deterministic parser) decomposes "What did Alice say about her trip after she got back?" into ⟨who: Alice⟩, ⟨what: trip⟩, ⟨temporal: after return⟩, retrieves for each, and joins. QDMR (Wolfson et al. 2020), Least-to-Most prompting (Zhou et al. 2022), Plan-and-Solve (Wang et al. 2023), Decomposed Prompting (Khot et al. 2022).

**Iterative retrieval with feedback.** Run retrieval → generate → check if answer is grounded → retrieve again if not. FLARE (Jiang et al. 2023) triggers retrieval on low-probability tokens. ITER-RETGEN alternates retrieval and generation rounds.

**Graph neural networks for retrieval.** GNNs over the document-citation graph or the entity-relation graph re-rank candidates by considering structural neighbors. Used in QA systems on WikiData, FB15k. Largely superseded by KG-traversal-plus-LLM in practice.

**Contrastive learning for retrieval.** The training objective behind all modern dense retrievers: pull (query, positive passage) together, push (query, negative passage) apart in embedding space. In-batch negatives (DPR), hard negative mining from BM25 (ANCE — Xiong et al. 2021), and distillation from cross-encoders (TAS-B, RocketQA, ColBERTv2) are the three workhorses.

**Dense Passage Retrieval (DPR) (Karpukhin et al., EMNLP 2020).** Two BERT bi-encoders, NLL loss with in-batch negatives plus one BM25-hard negative per question. On Natural Questions, TriviaQA, WebQuestions, CuratedTREC, DPR exceeded BM25 by 9–19 absolute points top-20 accuracy. Variants: ANCE, RocketQA, GTR, E5, BGE, GTE.

**REALM (Guu et al., ICML 2020).** End-to-end trains a retriever and a masked language model jointly; the retriever is updated by backpropagating through the retrieval step, with asynchronous re-indexing every few thousand steps. Outperformed prior open-QA models by 4–16 points absolute.

**RETRO (Borgeaud et al., DeepMind 2022).** Modifies the LM architecture itself with "chunked cross-attention" to a 2-trillion-token retrieval database; a 7B-parameter RETRO matches a 175B parameter GPT-style baseline. In-Context RALM (Ram et al.) is the architecture-free alternative that simply prepends retrieved documents to a frozen LM's input.

### 4. Unconventional Methods

**Retrieval by generation.** Generate candidate answers from the LM's parametric knowledge, then verify each against the DB. Inverts the usual retrieve-then-generate. Useful when the DB is small and structured (e.g., a personal calendar): generating "Tuesday 3pm" and confirming via `SELECT * FROM events WHERE date='2026-05-20' AND time='15:00'` is more reliable than vector-searching "what's on tuesday".

**Backward retrieval / generative retrieval.** Train a model to *generate the document identifier* directly from the query (DSI — Tay et al. 2022; NCI; RetroLLM, arXiv:2412.11919). The DocID is a hierarchical numeric code. *Good at:* end-to-end optimizable, no separate index. *Fails at:* hard to update the index without retraining.

**Structural matching.** Match the *parse tree* of the query against the parse tree of stored facts, not just the surface tokens. Example: query parses as `[ASK speaker=Alice, predicate=visited, object=?]`; only match stored facts with the same template. This is the deterministic core of QA-SRL (see below) and the recommended pattern for a deterministic reconstruction engine.

**Predicate-argument structure matching (QA-SRL).** He, Lewis & Zettlemoyer ("Question-Answer Driven Semantic Role Labeling," EMNLP 2015) represent predicate-argument structure as natural-language question-answer pairs: the verb "introduce" annotates with "What is introduced?" and "Who introduces something?" with the answer being a span. *Mechanically:* parse each stored utterance once at write-time into (verb, role, argument-span) tuples; at read-time, parse the question into the same schema and join on (verb, role). *Good at:* deterministic, interpretable, language-agnostic with cross-lingual QA-SRL (arXiv:2602.22865). *When to use:* as the *verification* stage of a deterministic engine.

**Abductive retrieval.** Given a question Q, retrieve facts F such that F → Q under some inference rule. Implemented in Datalog/Prolog-style systems and in some neuro-symbolic QA (DROP). *Good at:* numerical reasoning, counterfactuals. *Fails at:* requires explicit inference rules.

**Memory-augmented neural networks (NTM, DNC).** Neural Turing Machine (Graves et al. 2014) and Differentiable Neural Computer (Graves et al., *Nature* 2016) couple a recurrent controller to an external memory matrix with differentiable read/write heads. The DNC learned to answer family-tree questions and shortest-path on the London Underground map. Sparse DNC (Rae et al. 2016) and rsDNC (Franke, Niehues & Waibel 2018, arXiv:1807.02658) scaled to bAbI-style QA; MT-DNC (Liang et al. 2023) reaches 2.5% average WER on bAbI with only one failed subtask. *Status:* superseded by transformer-with-retrieval in practice but conceptually foundational; the "controller writes a fact then retrieves it later" pattern is the ancestor of every modern memory layer.

**Retrieval via program synthesis.** The LM generates a SQL/Cypher/Python query, which is executed deterministically against the DB. Text-to-SQL (Spider, BIRD benchmarks): Agentar-Scale-SQL (Wang et al., arXiv:2509.24403, September 2025) reports 81.67% execution accuracy on the BIRD test set, ranking #1 on the official leaderboard, against human experts at 92.96% on the same benchmark. *Good at:* exact aggregations, joins, temporal filters. *Fails at:* schema-comprehension errors. *When to use:* whenever your data is genuinely structured — never do RAG over a relational DB if you can do text-to-SQL.

**Schema-guided retrieval.** Constrain retrieval by typed slots (Schema-Guided Dialogue dataset, Google 2019). For a memory system, each memory has a type (preference, fact, event, relationship) and retrieval is gated by the type the question implies. Cognis uses 13 memory categories scoped by `owner_id + agent_id + session_id`.

**Temporal-aware retrieval.** (a) Time-decay weighting: score *= exp(-λ·age); (b) recency bucketing: separate indexes for last-7-days vs last-90-days vs archival; (c) bitemporal models (Zep/Graphiti): every edge carries (t_valid, t_invalid) intervals so retroactive corrections invalidate without deleting; (d) temporal filtering: explicit `WHERE date BETWEEN`. Cognis adds a "temporal boosting" multiplier on retrieval scores for time-sensitive queries.

**Entity-centric retrieval.** Instead of similarity over chunks, use entity linking to map the question to canonical entities, then retrieve all facts attached to those entities. KAPING (Baek et al. 2023) verbalizes N-hop triples around the question entity and embeds them. The Entity Retrieval paper (arXiv:2408.02795) showed that for entity-centric questions, sparse retrievers retrieve *less* relevant documents than entity-linking lookup. *When to use:* whenever the question contains a named entity, which for conversational memory is most of the time.

**Situation modeling for reconstruction.** Build a small mental model of the situation (Zwaan's event-indexing model: who, where, when, why, what) from scattered fragments, then answer from the model. Implemented in EMem (arXiv:2511.17208), which decomposes sessions into "elementary discourse units" — self-contained statements with normalized entities and source-turn attributions — and propagates them through a graph. The "Situation Calculus" (McCarthy) is the symbolic ancestor.

### 5. Banned/Dangerous Methods (And Why They Fail)

**Threshold-based scoring.** "Return results where cosine > 0.7." Fails because: (1) BM25 and cosine scores are not calibrated across corpora — a 0.7 in one domain is a 0.85 in another; (2) the same question against the same DB at different times can drift past the threshold as new content is added (changes IDF); (3) it converts a continuous ranking signal into a binary decision at the worst possible point (the inflection of the score distribution). *Use instead:* always rank, then apply a top-k cutoff with a *separate, structural* verification step.

**Token overlap as verification.** "If 60% of query tokens appear in the candidate, accept." Fails because LOCOMO adversarial questions are explicitly constructed (per Bini et al., 4 Dec 2025, arXiv:2510.23730) to reference "something that has high surface-level semantic similarity with part of the conversation" while having no actual answer. Token overlap rewards exactly the wrong signal — a stored sentence about Alice's last vacation will overlap heavily with a question about Alice's *next* vacation even when no answer exists.

**Self-verification loops.** Candidate answer is fed back to the same LM with the same retrieved context to "confirm" itself. Fails because: (1) the LM has already conditioned on this context once, so the marginal information gain is zero; (2) confirmation bias — autoregressive models systematically rate their own outputs higher than alternatives (Madaan et al. 2023, Stechly et al. 2024); (3) it amplifies hallucinations into "verified" hallucinations. *Use instead:* verification against a *different* representation (SQL on the structured store, predicate-argument template check, or a separately trained NLI model).

**Test contamination.** Training or caching on test questions, or letting test questions leak into the retrieval corpus. LOCOMO's QA pairs were generated by GPT-4 from event graphs; any system that subsequently fine-tunes on those event graphs is contaminated. The Penfield Labs audit of LOCOMO found 99 score-corrupting errors in 1,540 questions (6.4%), highlighting that even careful benchmarks have leakage and labeling bugs.

**Over-indexing on cosine similarity.** High cosine ≠ correct answer. Two failure modes: (1) embeddings collapse to surface lexical features when fine-tuned aggressively on a narrow domain, so cosine measures string overlap, not meaning; (2) cosine doesn't capture negation or quantification (the vector for "Alice did visit Paris" is close to "Alice did not visit Paris"). *Use instead:* cross-encoder reranking and NLI-based entailment checks.

**Greedy retrieval.** Taking the first match. The first BM25 match is often dominated by a single rare term (the "orangutan effect": one weird word in the query inflates a marginally relevant document to the top). Always retrieve k ≥ 20 and rerank.

**Retrieval without verification.** Returning the top-ranked passage as the answer. Fails because retrieval gives you *topical relevance*, not *factual answerability*. A passage about "Alice's trip" is topically relevant to "Where did Alice go?" but may not contain the destination. *Use instead:* always run a verification stage that confirms the candidate passage *contains a value of the right type for the question's wh-slot*.

### 6. Conversational Memory QA Systems (Specific to LOCOMO-style Benchmarks)

**Mem0 (Chhikara et al., ECAI 2025, arXiv:2504.19413).** Two-phase pipeline: *Extraction* — LLM (GPT-4o-mini in the paper) reads the last M=10 messages plus a running summary and emits candidate facts. *Update* — for each fact, retrieve top S=10 semantically similar existing memories via dense embeddings; function-call the LLM to choose ADD / UPDATE / DELETE / NOOP. Storage: Qdrant/Pinecone/pgvector etc. for vectors; Neo4j for the optional Mem0^g graph variant, which extracts entities and stores both an entity-relationship triple view and a semantic-triplet view (encoding the whole query as a vector and matching against triplet text encodings with a configurable relevance threshold). LOCOMO results in their paper: single-hop F1 = 38.72, BLEU-1 = 27.13, J (LLM-judge) = 67.13 — *with the adversarial category explicitly excluded*.

**Zep / Graphiti (Rasmussen et al., arXiv:2501.13956).** Memory is a temporally-aware knowledge graph G = (N, E, φ) with three hierarchical subgraphs: *episode* (raw events), *semantic entity* (extracted entities), *community* (clusters). Every edge has explicit (t_valid, t_invalid) intervals and a bitemporal model that separates event time T from ingestion time T'. Retrieval is *hybrid*: semantic embeddings + keyword search + graph traversal, in near-constant time independent of graph scale. On the Deep Memory Retrieval (DMR) benchmark from the MemGPT team, Zep posts 94.8% vs MemGPT's 93.4%; on LongMemEval, claimed improvements up to 18.5% with 90% lower response latency. The methodology of these numbers vs LOCOMO is disputed (see below).

**Cognis (Daftari et al., Lyzr, arXiv:2604.19771).** The closest published instantiation of the user's desired architecture. Pipeline (verbatim from the abstract): "Cognis combines a dual-store backend pairing OpenSearch BM25 keyword matching with Matryoshka vector similarity search, fused via Reciprocal Rank Fusion. Its context-aware ingestion pipeline retrieves existing memories before extraction, enabling intelligent version tracking that preserves full memory history while keeping the store consistent. Temporal boosting enhances time-sensitive queries, and a BGE-2 cross-encoder reranker refines final result quality." Concrete settings: 256-D Matryoshka shortlist → 768-D rerank; RRF at 70% vector / 30% BM25 (tuned by ablation); open-source build replaces OpenSearch + Matryoshka with SQLite FTS5 + Qdrant local; 13 memory categories scoped by `owner_id + agent_id + session_id`. Reports state-of-the-art on both LOCOMO and LongMemEval across eight answer-generation models.

**MemGPT / Letta (Packer et al., arXiv:2310.08560).** Three memory tiers, OS-inspired:
- *Core/main memory*: a fixed-size editable section of the LLM context (human-readable, agent-editable scratchpad for persona and key facts).
- *Recall memory*: full conversational log, searchable via tool calls.
- *Archival memory*: external DB (typically pgvector), agent reads/writes via tool calls.
Function calling drives the paging: when the context fills past a threshold, the agent emits a `core_memory_replace` or `archival_memory_insert` call, evicts the oldest messages, and a summarizer creates a compressed running summary that is reinjected at the front. The Letta framework (formerly MemGPT, after the commercial spin-out) is the production implementation. *Strength:* unbounded effective context. *Weakness:* relies on the agent to manage its own memory — performance is bottlenecked on the LLM's tool-use reliability.

**MemMachine (arXiv:2604.04853).** Episodic Memory system reporting 0.8487 overall on LoCoMo and 0.9169 with gpt-4.1-mini per their March 2026 blog, positioning above Mem0, Zep, Memobase, and LangMem on the same benchmark.

**A-MEM (Xu et al., 2025, arXiv:2502.12110).** Agentic memory with note generation, link generation, and memory evolution modules; evaluated across all 5 LOCOMO categories including adversarial; Table 7 of the paper is the authoritative source for per-category numbers across MemGPT, Mem0, A-MEM, etc.

**Other LOCOMO-tier systems worth knowing:** MemoryBank (Zhong et al. 2024), ReadAgent (Lee et al. 2024), RMM (Reflective Memory Management, arXiv:2503.08026, dense retriever + lightweight MLP reranker, >5% over strongest baseline on MSC/LongMemEval), EMem (event-centric, source-turn attribution), ENGRAM (typed dense retrieval, arXiv:2511.12960, claimed SOTA on LOCOMO), EverMemOS (agentic multi-round BM25+vector+exact match, 92.3% on LoCoMo per arXiv:2603.15599), and LiCoMemory (73.8% accuracy / 76.6% recall on LongMemEval, Huang et al., 3 Nov 2025).

**Multi-hop reasoning over conversation history.** Two paradigms: (a) *Plan-then-execute*: decompose the question into a sub-question DAG (QDMR, Decomposed Prompting), retrieve evidence per sub-question, join. (b) *Iterative retrieve-reason*: IRCoT, FLARE, ITER-RETGEN. PRISM (arXiv:2510.14278) and PAR-RAG use a Question Analyzer → Retriever → Selector → Synthesizer pipeline with verifier feedback. For LOCOMO specifically, multi-hop accuracy without entity-linking collapses badly; entity-linking-plus-graph-traversal (Zep, Mem0^g, EMem) substantially outperforms pure vector RAG on multi-hop.

**Adversarial refusal — knowing when NOT to answer.** This is the dimension where most LOCOMO leaderboard systems silently cheat. Maharana et al. (2024) report LLM F1 ≈ 2.1 vs human 89.4 on adversarial. The Mem0 paper text reads: "The dataset originally included an adversarial question category, which was designed to test systems' ability to recognize unanswerable questions" — and then evaluates without it. The Zep–Mem0 GitHub dispute (zep-papers issue #5, May 2025) hinged on this: Mem0's CTO showed that with adversarial removed, Zep's mean accuracy drops to 58.44% ± 0.20 (from a reported 65.99% ± 0.16); Zep countered with 75.14% ± 0.17 in a blog rebuttal. The ATANT v1.1 audit (arXiv:2604.10981) found that for 444 of 446 category-5 questions, gold answer is empty, and the scoring script "returns False regardless of the prediction" when gold is empty, so "a system that correctly refuses (returns empty or 'I don't know') is scored identically to a system that fabricates." Recommended techniques for abstention:
- *PassiveQA* (arXiv:2604.04565): three-action framework Answer/Ask/Abstain with a supervised planner; reports macro F1 55.6% (+20.3 pp), Abstain recall 13.3% → 58.1%, hallucination rate 42.7% → 33.8%.
- *Linear Directions for Unanswerability* (arXiv:2509.22449): learns an "unanswerability direction" in LLM activation space; generalizes better than prompt/classifier baselines on Llama-3-8B-Instruct and Gemma-3-12B-IT.
- *AbstentionBench* (Kirichenko et al. 2025): 20 datasets, six abstention scenarios.
- *CoCoNot* (Brahman et al. 2024): taxonomy — unsafe / unsupported / indeterminate / incomprehensible.
- *RefusalBench* (arXiv:2510.10390): 176 linguistic levers for generative refusal evaluation.
- *ConvoMem* (arXiv:2511.10523): explicit abstention scoring that "inverts the typical validation logic, recognizing 'I don't know' responses as correct while treating any specific answer as failure"; 75,336 QA pairs.
- *LongMemEval* (Wu et al., ICLR 2025): an "I don't know"-only baseline scores 5.8% accuracy, defining the floor for abstention-aware evaluation.

For a deterministic reconstruction engine, the right abstention discipline is: **a verification stage that returns "no answer" if no stored fact has a value of the required wh-slot type**. Don't rely on confidence thresholds.

**Temporal QA over dialogue.** Three layers: (a) extract explicit times at write-time and store them in a `datetime` column; (b) maintain (t_valid, t_invalid) for facts that change (Zep's bitemporal model); (c) for relative time expressions ("after she got back"), resolve at parse time using SUTime, HeidelTime, or DUCKLING into absolute intervals. LOCOMO's temporal category drops models hard: GPT-3.5-turbo-16K scores 20.3 F1 vs human 92.6 F1 (Maharana et al., ACL 2024). The gap is mostly resolution of relative expressions and ordering across sessions.

**Speaker attribution in multi-speaker conversations.** Often the silent failure. EverMemBench (arXiv:2602.01313) finds multi-hop accuracy collapses to 26% under multi-party attribution even with oracle evidence: "conversations are often multi-party, requiring the system to track who said what and how information propagates across speakers and groups." GroupMemBench (arXiv:2605.14498) targets "(i) group dynamics that go beyond concatenated one-on-one chats, (ii) speaker-grounded belief tracking, where the per-user memory modeling is needed, and (iii) audience-adapted language." SA-LLM (arXiv:2503.08842) uses contrastive speaker-aware learning with speaker-attributed input encoding. Adobe Research's text-based speaker identification (arXiv:2407.12094) uses pretrained LMs to classify utterances by speaker. The deterministic recipe is to (a) prepend speaker tags to every utterance at index time, (b) store speaker as a *column*, not a textual prefix, (c) require speaker match as an SQL filter before similarity ranking. Maharana et al. (2024) themselves list "inaccurate speaker attributions" as one of five named error categories on LOCOMO's event-graph task.

### 7. Deterministic / Non-LLM Approaches

This is the section most relevant to your stated goal: a deterministic reconstruction engine for conversational memory.

**Retrieval without any LLM call.** The full pipeline:
1. **Write-time extraction** (the one LLM-permitted step, or alternatively rule-based with spaCy/Stanza). Parse each utterance into (speaker, timestamp, predicate, arguments, named entities, sentiment). Persist as rows.
2. **Storage**: SQLite with FTS5 for text + B-tree indexes on (entity, speaker, timestamp, predicate). Optionally pgvector or Qdrant local for an embedding column.
3. **Query-time parsing**: dependency parse with spaCy or Stanza; extract the question's wh-word, predicate (root verb + lemma), arguments, entities, and time constraints.
4. **Retrieval**: (a) SQL filter on speaker/time/entity from the parse; (b) FTS5 BM25 over the filtered subset; (c) optional embedding cosine over the filtered subset; (d) RRF fuse; (e) deterministic reranker — score = α·BM25_rank + β·cosine_rank + γ·entity_overlap + δ·predicate_match + ε·recency.
5. **Verification**: confirm the candidate passage contains a value of the required wh-slot type (NER + type-checking). If not, return "I don't know" — and *mean it*, not as a confidence threshold but as a structural assertion.
6. **Surface generation** (optional LLM call): only at the very end, paraphrase the verified answer span into a natural sentence. Or skip this and return the extracted span directly.

This is the architecture under Cognis's open-source build (FTS5 + Qdrant) and the recommended starting point for the user's engine.

**Structural/grammatical approaches to QA.**
- *Dependency parsing*: spaCy (en_core_web_trf), Stanza, UDPipe. Extract subject, object, root verb, modifiers.
- *Semantic role labeling*: AllenNLP's BERT-SRL, PropBank-style (A0 = causer, A1 = patient, A2 = destination, AM-TMP = temporal modifier, AM-LOC = locative). For each predicate, you get a typed argument structure.
- *QA-SRL* (He, Lewis, Zettlemoyer EMNLP 2015): represents roles as natural-language questions, which removes the dependency on PropBank ontologies and is annotator-friendly. Cross-lingual QA-SRL annotation extends this without language-specific lexica (arXiv:2602.22865).
- *AMR / UDS*: more abstract semantic graphs. Heavier but more expressive.
*Recipe:* parse every stored utterance once into (predicate, A0, A1, A2, AM-TMP, AM-LOC); store as a JSON column. At query time, parse the question into the same schema and join.

**Rule-based reconstruction from structured databases.**
- Datalog/Prolog rules over entity-relation triples for inference ("if A is parent of B and B is parent of C, then A is grandparent of C").
- AKBC (Automatic Knowledge Base Construction) style: extract triples, store in RDF or Neo4j, query with SPARQL/Cypher.
- Best for: family/organization relationships, preferences, factual claims.

**SQL-based fact verification.**
- For every candidate answer the retriever produces, generate the SQL that would *confirm* it: `SELECT COUNT(*) FROM facts WHERE entity='Alice' AND predicate='visited' AND object='Paris' AND date BETWEEN ? AND ?`.
- If the count > 0, the answer is grounded; if 0, abstain.
- This is the deterministic analog of self-consistency, with the LLM replaced by SQL.
- Implementation tip: store derived predicates separately (e.g., `is_user_preference`, `is_factual_claim`, `is_temporal_event`) so verification can be scoped by predicate type.

**Achieving high recall AND high precision without probabilistic models.**
The classical IR trick is *recall via cheap broad retrieval, precision via expensive narrow verification*. For a deterministic engine:
- *Recall stage*: union of (a) entity-based SQL lookup, (b) FTS5 BM25, (c) sparse-dense fusion if you allow embeddings. Cast a wide net (k=50–200).
- *Precision stage*: predicate-argument template matching from the question's parse against each candidate's stored parse. Drop candidates that fail to match the question's argument types.
- *Abstain*: if no candidate survives precision, return "no answer." This is what produces the high precision; the broad recall stage produces the high recall.

The benchmark this approach should beat on LOCOMO single-hop, temporal, and adversarial — categories where structural verification is most valuable — is well-tuned hybrid RAG with cross-encoder. Multi-hop and open-domain are where dense retrieval still wins because the inference required exceeds what a parse-and-join pipeline covers.

---

## Recommendations

**Stage 1 — Get to LOCOMO-baseline parity with deterministic core (target: ~70% single-hop, ~60% temporal, ~80% adversarial-as-refusal).**
1. SQLite + FTS5 (porter tokenizer) for text; columns for `speaker`, `session_id`, `timestamp`, `entity_list`, `predicate`, `wh_type_supported`.
2. At write-time, parse each utterance with spaCy or Stanza into predicate-argument tuples; store JSON.
3. At read-time, parse the question; filter by speaker/time/entity in SQL; rank surviving rows with FTS5 BM25.
4. Verify: confirm the top candidate contains a span of the right NER type for the question's wh-word. If no — abstain.
5. **Threshold to change recommendation:** if FTS5 BM25 alone achieves <60% single-hop on your dataset, add the embedding channel.

**Stage 2 — Add dense channel and RRF (target: +5–10 points single-hop, +10 points open-domain).**
1. Add a Qdrant local or pgvector column with a strong open-weight embedder (`bge-large-en-v1.5` or `gte-large`).
2. RRF-fuse FTS5 rank with cosine rank at k=60.
3. Keep the structural verification stage unchanged.
4. **Threshold to change recommendation:** if dense retrieval is adding noise (recall improves but precision tanks), tune to weighted RRF or drop dense for short queries.

**Stage 3 — Add a cross-encoder reranker (target: +5 points overall, mostly precision@1).**
1. Use `BAAI/bge-reranker-large` or `BGE-2` (Cognis's choice) over the top 50 from Stage 2.
2. Latency cost: 50–200ms per query. Acceptable for memory QA; if not, distill to a smaller reranker.
3. **Threshold to change recommendation:** if reranker is changing top-1 in <10% of queries, your first stage is too narrow — increase k.

**Stage 4 — Add abstention discipline (target: +30 points adversarial).**
1. Implement structural abstention: if no candidate has a span of the required wh-type after verification, return "I don't know."
2. Add an "unanswerability direction" probe (arXiv:2509.22449) if you allow a small LLM in the loop.
3. Evaluate against ConvoMem's abstention-aware scoring, not LOCOMO's broken category-5 script.
4. **Threshold to change recommendation:** if your abstention rate is <5% or >40%, your verification is mis-calibrated; tune the wh-type-match rules.

**Stage 5 — Add temporal and speaker awareness as first-class columns (target: +10–15 points temporal, multi-speaker robustness).**
1. Extract explicit times at write-time using SUTime/DUCKLING; store absolute timestamps.
2. Bitemporal: store both event-time and ingestion-time per row.
3. Speaker is always a column filter, never a text prefix. Multi-speaker mode requires per-speaker memory shards.
4. **Threshold to change recommendation:** if your speaker-attribution error rate exceeds the LOCOMO ground-truth error rate of ~1.5% (24/1540), your extraction is the bottleneck, not retrieval.

**Stage 6 — Add a knowledge-graph layer for multi-hop only if needed.**
1. Extract (subject, predicate, object) triples at write-time; store in a graph table (id, src, dst, predicate, t_valid, t_invalid).
2. For multi-hop questions only, run a bounded BFS (≤3 hops) from question entities.
3. Use Cypher (if Neo4j) or recursive CTEs (if SQLite).
4. **Threshold to change recommendation:** if multi-hop accuracy is <30%, you need the graph; above that, the parse-and-join pipeline is probably enough.

---

## Caveats

- **The benchmark is broken in places.** LOCOMO's "adversarial" category 5 has 444/446 questions with empty gold and a scoring script that returns False regardless of prediction. The Penfield Labs audit found 99 corruption errors in 1,540 questions, including 24 speaker-misattribution errors. Treat any single LOCOMO number with at least ±5 point skepticism, and validate on a held-out internal dataset.
- **The Zep–Mem0 dispute is unresolved as of May 2026.** Different parties report different numbers under different evaluation rules; numbers in this report follow the explicit citations and the user should not assume cross-paper comparability.
- **The Cognis paper (arXiv:2604.19771) is a recent preprint** — it has not been independently replicated outside Lyzr's own evaluation as of writing. Its architecture is sound on first principles and matches what works in other hybrid systems, but the specific SOTA claims should be reproduced before being relied on for a production decision.
- **Some banned methods are useful in restricted settings.** Threshold scoring works fine when you can calibrate per-corpus; token overlap is a perfectly reasonable signal for *recall* (just not for verification); self-verification loops help when the verifier sees different evidence than the candidate. The danger is using them naively as primary decision mechanisms.
- **Several papers cited (arXiv IDs starting 2602–2605) are 2026-vintage preprints** with limited peer review. They represent the current research frontier rather than established results.
- **Deterministic reconstruction does not eliminate LLMs entirely.** Write-time extraction and surface paraphrase still benefit from LLM use; the deterministic property is about the *retrieval and verification* path, where the LLM is the largest source of non-reproducibility. A purely rule-based extraction pipeline (spaCy + custom rules) is feasible but typically gives up 10–20 points on extraction quality.
- **The "right" architecture depends on your read/write ratio.** If reads dominate (typical for conversational memory), expensive write-time extraction is amortized and the deterministic engine wins on latency and reliability. If writes dominate (high-velocity ingestion), you may need to defer extraction and accept lazier indexes.