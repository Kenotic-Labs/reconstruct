# Grammar Engine — Formal English Rules

Every rule here is from linguistics (Quirk et al., Cambridge Grammar, formal syntax).
Every rule maps to spaCy dep/POS/morph tags. No "I think" — only rules.

## 1. EPISODIC (what happened)

**Rule:** English clause = Subject + Predicate. Predicate is everything that is NOT the subject subtree.

**Subject deps:** nsubj, nsubjpass, csubj, csubjpass, expl
**Episodic fact = all tokens - subject subtree - leading auxiliaries**

Cases:
- Simple: "Caroline [has been researching agencies]" — remove nsubj subtree
- Compound subject: "Caroline and Mark [went home]" — remove nsubj + conj subtrees
- Expletive: "There [are three dogs]" — remove expl
- Passive: "The book [was read by John]" — remove nsubjpass subtree
- Imperative: no subject — entire clause is predicate

## 2. EMOTIONAL (how it felt)

**Rule:** English encodes emotion through predicate adjectives after copular verbs.

**Copular verbs:** be, seem, become, appear, feel, get, go, grow, keep, look, remain, smell, sound, stay, taste, turn

**Emotion positions in spaCy:**
- dep=acomp (adjectival complement): "She is **happy**"
- dep=attr when POS=ADJ: "She became **anxious**"
- dep=advmod on emotion adverb: "She spoke **angrily**"
- ROOT is emotion verb: "She **loves** him"

**Valence:** SentiWordNet or WordNet noun.feeling supersense. Syntactic negation (dep=neg) inverts valence.

**Target:** prep child of the emotion word = target. "angry **about the news**" → target = "the news"

## 3. TEMPORAL (when)

**Rule:** English has 2 morphological tenses + periphrastic future.

**Direction from verb morphology:**
- VBD or morph Tense=Past → past
- VBZ/VBP or morph Tense=Pres → present
- MD with lemma will/shall → future
- "going to" pattern → future

**Expression from temporal adverbials (5 forms):**
1. Adverb: dep=advmod, tag=RB — "yesterday", "now", "soon"
2. Prep phrase: dep=prep with temporal pobj — "on Tuesday", "in March"
3. Bare NP: dep=npadvmod — "last week", "next year", "four years ago"
4. Subordinate clause: dep=advcl with mark=when/while/before/after/since/until
5. NER: ent_type_ in (DATE, TIME)

## 4. RELATIONAL (who's involved)

**Rule:** English marks participants through grammatical relations.

**Participant deps:**
| dep | Role | Example |
|-----|------|---------|
| nsubj | Subject (active) | "**Caroline** told Melanie" |
| nsubjpass | Subject (passive) | "**Melanie** was told" |
| dobj | Direct object | "told **Melanie**" |
| iobj/dative | Indirect object | "gave **Melanie** a book" |
| pobj under agent | Agent in passive | "told by **Caroline**" |
| pobj under prep | Oblique | "talked with **Melanie**" |
| poss | Possessive | "**her** book" |
| appos | Appositive | "Caroline, **my friend**" |
| conj | Conjunct | "Caroline **and Melanie**" |

**Entity filter:** ent_type_=PERSON, pos_=PROPN, pos_=PRON

## 5. SCHEMATIC (what category)

**Rule:** No syntactic rule. This is semantic. Use WordNet supersenses.

**Method:** Get supersense of ROOT verb's dobj (or ROOT verb if intransitive).
- nltk.corpus.wordnet.synsets(word).lexname() → "noun.act", "verb.cognition", etc.

**45 WordNet supersenses** map to schema categories:
- verb.cognition → career/education
- noun.location → housing
- noun.person/noun.group → social/family
- verb.emotion/noun.feeling → emotional
- noun.food/verb.consumption → health
- verb.creation → hobby
- noun.possession/verb.possession → finance
- noun.time → temporal

## 6. SENTENCE CLASSIFICATION

**4 types, 4 syntactic rules:**

| Type | Rule | spaCy |
|------|------|-------|
| Interrogative | Has "?" OR aux before nsubj OR WH-initial + SAI | sent ends with "?", or first token dep=aux before nsubj |
| Imperative | Base verb (VB) + no nsubj | ROOT.tag_=="VB" and no nsubj child |
| Exclamatory | "What a" / "How" + ADJ + "!" | WH-initial + no SAI + ends "!" |
| Declarative | Everything else | Default |

## 7. MOOD

**4 values:**

| Mood | Rule | spaCy |
|------|------|-------|
| Indicative | Default | morph Mood=Ind or no special markers |
| Imperative | VB + no subject | ROOT.tag_=="VB", no nsubj, morph Mood=Imp |
| Subjunctive | Bare stem after demand/suggest verbs, "were" for all persons | VB in ccomp with mark="that" after suggest/demand; "were" with 1st/3rd sg |
| Conditional | would/could/should/might + verb | aux with lemma in (would, could, should, might) |

## 8. TENSE-ASPECT (12-cell grid)

| | Simple | Progressive | Perfect | Perfect Progressive |
|---|---|---|---|---|
| Past | VBD, no aux | was/were(aux) + VBG | had(aux) + VBN | had(aux) + been(aux) + VBG |
| Present | VBZ/VBP, no aux | am/is/are(aux) + VBG | have/has(aux) + VBN | have/has(aux) + been(aux) + VBG |
| Future | will(MD) + VB | will(MD) + be(aux) + VBG | will(MD) + have(aux) + VBN | will(MD) + have(aux) + been(aux) + VBG |

**Tense from first finite verb:** VBD→past, VBZ/VBP→present, MD(will)→future
**Aspect from aux chain:** no aux→simple, be+VBG→progressive, have+VBN→perfect, have+been+VBG→perfect progressive

## 9. VOICE

**Rule:** Passive = be + VBN. Active = everything else.

**spaCy:** `any(tok.dep_ in ("auxpass", "nsubjpass") for tok in doc)` → passive. Else active.

**Agent:** dep=agent → the "by" phrase. "bitten **by the dog**"

## 10. PRONOUNS

**Resolution rules:**
| Person | Pronouns | Resolves to |
|--------|----------|-------------|
| 1st | I, me, my, mine, myself, we, us, our, ours, ourselves | Speaker |
| 2nd | you, your, yours, yourself, yourselves | Listener |
| 3rd | he, him, his, she, her, it, they, them, their | Referenced entity (needs coreference) |

**spaCy morph:** Person=[1]→speaker, Person=[2]→listener, Person=[3]→entity
