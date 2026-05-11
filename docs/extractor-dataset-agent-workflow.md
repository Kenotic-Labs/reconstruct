# Extractor Dataset Agent Workflow

This repo now contains four purpose-built agents for building the English-only `Nura Extractor v7` dataset from internet-sourced English conversation.

## Agents

- `.claude/agents/extractor-dataset-manager.md`
- `.claude/agents/web-conversation-miner.md`
- `.claude/agents/extractor-schema-writer.md`
- `.claude/agents/temporal-affect-curator.md`

## Recommended workflow

1. Start with `extractor-dataset-manager`
   - define the current predicate families, source targets, and release scope

2. Run `web-conversation-miner`
   - search the internet for short English utterances
   - record provenance
   - write raw source inventories

3. Run `extractor-schema-writer`
   - convert mined utterances into the master schema
   - derive cleanup, triplets, roles, and affect task files

4. Run `temporal-affect-curator`
   - focus on durations, dates, historical framing, corrections, and explicit emotions

5. Hand back to `extractor-dataset-manager`
   - audit coverage
   - freeze train/eval/test splits
   - publish a dataset release report

## Intended directories

```text
data/extractor_v7/
  source_inventory/
  raw_web/
  master/
  cleanup/
  triplets/
  roles/
  affect/
  reports/
```

## Scope

These agents are designed for:

- English only
- unseen real-world English conversation
- extractor dataset production

They are not intended to modify production code in `app/`.
