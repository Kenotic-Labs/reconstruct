"""One-shot script to write the pricing addendum. Delete after use."""
import pathlib

content = r"""# PRD v2 Addendum — Pricing Model
## Per Continuity Instance Pricing

**Date:** 2026-04-26
**Author:** Samuel Sameer Tanguturi
**Addendum to:** PRD_Continuity_Architecture_v2.md (Section 13)
**Research basis:** Market pricing analysis (Mem0, Zep, Letta, Pinecone, Supabase, GitHub Copilot), SaaS conversion benchmarks, developer willingness-to-pay surveys (Heavybit, indie hacker data), habituation/stickiness research (Kahneman loss aversion, Slack retention data, GitHub Copilot dependency studies, Flexera switching cost surveys)

---

## The Unit: One Living Situation Store

Every continuity instance is a real, separate situation store — its own SQLite file, its own traces, its own edges, its own reconstruction. Pricing maps to real resource allocation, not artificial feature gating. Every tier gets full 7-property continuity. No features locked behind higher tiers.

---

## Pricing Structure

| Unit | Price | What it is |
|------|-------|-----------|
| Developer | $50/year | 1 dev account. Access to SDK/MCP. Can build and test. |
| Agent | $50/year each | Each agent the dev builds gets its own continuity. |
| User | $50/year each | Each end-user gets their own continuity. Each person is a separate living state. |

---

## How It Scales

| Customer | Devs | Agents | Users | Annual |
|----------|------|--------|-------|--------|
| Solo indie hacker | 1 | 1 | 10 | $600 |
| Small startup | 3 | 5 | 1,000 | $50,650 |
| Mid-market | 10 | 20 | 10,000 | $501,500 |
| Enterprise | 50 | 50 | 100,000 | $5,005,000 |

---

## Why This Works

**No feature gating.** The indie hacker with 1 agent and 10 users gets the SAME 7-property architecture as the enterprise with 100K users.

**Near-zero COGS.** Everything runs on the customer's hardware. Kenotic hosts nothing. Serves nothing. Stores nothing. Pure licensing margin.

**Linear with customer growth.** More users = more continuity instances = more revenue. Aligned incentives.

**Competitive context (from market research, April 2026):**
- Mem0: $19-249/month, feature-gated (graph locked behind $249/mo)
- Zep: $25-475/month, no feature gating (pay for volume)
- Letta: $20+/month cloud, free self-hosted
- GitHub Copilot: $10/month (the proven individual dev price anchor)
- Kenotic: $50/year per instance (~$4.17/month) — full 7-property continuity

Different category — they sell memory. Kenotic sells continuity.

---

## Trial Model: 14-21 Day Living Trial

### Why not follow "industry standard" trial lengths?

There is no enforcer. No authority that dictates trial length. Industry benchmarks are data points, not laws. Kenotic defines its own category and its own rules. The trial model is designed from first principles around the product's unique property: accumulated living state.

### How the trial works

**During trial (14-21 days):**
- Full continuity. No limits. No feature gating.
- Every conversation builds the situation store. Traces accumulate. Supersession works. Proactive arcs form. Reconstruction improves with every interaction.
- The developer experiences what continuity feels like at full power.

**After trial expires:**
- Everything already stored STAYS. Nothing is deleted. Nothing is locked. Nothing is held hostage.
- The developer can still query, retrieve, reconstruct from everything already in the situation store. The accumulated understanding is theirs. It keeps working.
- **But new conversations stop getting continuity.** No new ingestion. No new traces. No new supersession. No new arcs. The situation store freezes in time.
- The AI still knows what it knew. But it stops learning. It stops updating. It stops following up. It stops growing.

**To reactivate:**
- Pay $50/year per instance. Continuity resumes. New conversations build on everything already there. No gap. No data loss. Just growth resuming.

### Why this model, not the alternatives

**Why not kill the data after trial (hard cutoff)?**
Gartner research: 58% of customers who feel "trapped" by a vendor eventually leave and become detractors. Deleting accumulated context after a trial feels punitive. The lock-in should feel like value, not a cage.

**Why not unlimited free tier (Supabase/Neon model)?**
Usage-limited free tiers work for databases where the upgrade trigger is storage/compute ceilings. For continuity, the value IS the accumulated state — there's no natural ceiling that triggers upgrade. A free tier risks indefinite free usage with no conversion trigger.

**Why the freeze model is better than both:**
- No data hostage — the developer keeps everything. Trust preserved.
- The contrast is visceral — for 2-3 weeks, every conversation builds understanding. Then it stops. The developer feels the difference immediately. Not "we took something away." Just "it stopped growing."
- The endowment effect is preserved — the data is there, they OWN it, they just need to pay to keep it alive.
- Natural conversion trigger — the moment the developer has a new conversation and the AI doesn't learn from it, they feel the gap. That's the conversion moment.

---

## The Science of Stickiness — Why This Product Is Uniquely Retentive

### Loss Aversion (Kahneman & Tversky, Prospect Theory)

The negative utility from losing something is approximately 2x the positive utility from gaining something equivalent. After 2-3 weeks of accumulated continuity:

The developer is NOT evaluating: "Is continuity worth $50/year?"
The developer IS evaluating: "Is losing all this accumulated context worth saving $50/year?"

Fundamentally different calculation. The $50 feels trivial against weeks of accumulated understanding.

### The Endowment Effect

Research shows trial users who performed 3 "ownership actions" within 48 hours converted at 340% higher rates than average. Full habit formation takes 66 days (median, Lally et al. 2010 UCL study). But you don't need full habit formation. You need the endowment effect — people assigning higher value to things they already possess. That activates within days if engagement is high.

With a continuity SDK, every store() call is an ownership action. Every conversation that gets ingested is the developer investing their data into the product. By day 3, the endowment effect is active. By day 14, it's entrenched.

### Three Types of Switching Costs (Burnham et al. taxonomy)

1. **Procedural** — time/effort to rebuild. "I'd have to re-explain everything to my AI."
2. **Financial** — sunk investment. "I've spent 3 weeks building this context."
3. **Relational** — emotional discomfort. "The AI knows my life. Starting over feels like amnesia."

Most developer tools only create Type 1 switching costs (config files, deployment setup, schema). The continuity SDK creates all 3, with Type 3 (relational) being the strongest.

### Real Product Evidence

**Slack:** Teams that hit 2,000 messages have 93% retention. The message history IS the product. Organizations with 80%+ employee adoption show 62% lower likelihood of switching.

**GitHub Copilot:** 88% code retention rate. 46% of all code is Copilot-generated for active users. Developers who disable it after months report being unable to write basic syntax without it. Kenotic creates the same behavioral dependency PLUS accumulated data. Stronger lock-in than Copilot alone.

**Flexera 2023:** 47% of enterprises cite data migration as the single most significant barrier to switching providers.

**McKinsey:** B2B companies with strong lock-in strategies achieve 13% higher revenue growth compared to industry peers.

### The SDK's Unique Advantage

Unlike most developer tools where switching means losing configuration/deployment setup, switching away from a continuity SDK means the AI forgets the user's life. That's not a config file you can export. It's not a schema you can migrate. It's accumulated understanding — the traces of how someone's situation evolved, what changed, what matters now, what the system learned to follow up on.

The switching cost is not procedural (re-entering data). It's relational (the AI forgetting everything). That's the strongest form of lock-in in the academic switching cost taxonomy. And it happens naturally through use — not through deliberate trapping, not through data hostage, not through punitive deletion.

The product physics create the stickiness. The pricing just lets the developer choose when to commit.

### Churn Benchmarks

Products with deep accumulated state consistently show lower churn:
- Enterprise SaaS (deep integrations, high stored state): annual churn 3-5%
- SMB SaaS (shallow integrations, low stored state): annual churn 10-15%+
- Customers paying >$250/month (deeper integration): lowest churn across all benchmarks

### The Critical Warning (Gartner)

58% of customers who feel "trapped" by a vendor eventually leave and become detractors. The lock-in MUST feel like value, not a cage. This is why:
- Data is never deleted after trial
- Existing queries keep working after trial
- Only new ingestion stops
- The developer always owns their data
- Reactivation is instant — pay and growth resumes

The stickiness comes from the product being genuinely valuable, not from making it painful to leave. Serve, not command.

---

## Revenue Milestones

| Milestone | What it takes |
|-----------|--------------|
| $100K ARR | ~170 solo devs with small apps (10 users each) |
| $500K ARR | ~10 startups with 1K users each |
| $1M ARR | ~20 startups with 1K users each, or 2 mid-market customers |
| $5M ARR | 1 enterprise customer, or 10 mid-market customers |

---

## Why $50/Year Per Instance Is Defensible

1. **Each instance is a real resource.** Separate SQLite file, separate traces, separate reconstruction. Not artificial scarcity.

2. **The value compounds over time.** Day 1: the instance knows nothing. Day 100: it knows the user's entire context. Day 1000: it knows the arc of their life.

3. **Identity stickiness.** The situation store IS the user's accumulated understanding. Leaving means losing a part of their operational self.

4. **Network effects within organizations.** If a company's 5 agents all have continuity, and their 1,000 users all have continuity, switching means losing years of accumulated institutional context.

5. **ATANT as procurement requirement.** When "ATANT-compliant" becomes a checkbox (like SOC 2 for security), vendors need to certify. Kenotic defined the standard. Kenotic passes the standard. Everyone else has to catch up.

---

## Research Sources

- Kahneman & Tversky — Prospect Theory (loss aversion ~2x gain)
- Lally et al. 2010, UCL — Habit formation median 66 days (European Journal of Social Psychology)
- Nir Eyal — "Hooked" model, Investment Phase (stored value = retention)
- Slack growth data — 2,000 messages = 93% retention threshold
- GitHub Copilot — 88% code retention, 46% code generation rate, 20M+ users
- Flexera 2023 State of the Cloud — 47% cite data migration as #1 switching barrier
- McKinsey — 13% higher revenue growth with strong lock-in strategies
- Gartner — 58% of "trapped" customers eventually leave (the cage vs value distinction)
- Burnham et al. — Consumer Switching Costs typology (procedural/financial/relational)
- SaaS conversion benchmarks — 14-day trials convert 71% better than 30-day
- Endowment effect research — 3 ownership actions = 340% higher conversion
- Mem0 pricing ($19-249/mo), Zep ($25-475/mo), Letta ($20+/mo), GitHub Copilot ($10/mo)
- Enterprise SaaS churn: 3-5% annual. SMB: 10-15%+
"""

pathlib.Path(r"D:\Nura\Kenotic Labs\PRD_v2_Addendum_Pricing.md").write_text(content, encoding="utf-8")
print("Written successfully:", len(content), "chars")
