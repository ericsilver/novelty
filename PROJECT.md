# Project: Trademark-driven innovation by industry

- Origin: imported from claude.ai project `019d3c7c-48ed-77d4-8c79-98c275e24df9` (created 2026-03-30).
- Default mode when working in this folder: `/explorer` (see `~/.claude/commands/explorer.md`). Map the literature first; Eric is a naif on the lit even though he is far from a naif on the methods. Critique is opt-in via `/critic`.

## Status (as of 2026-05-24)

Project scaffold exists; nothing substantive has been pulled in yet. Eric has work on this across the four-machine fleet (see `INBOX.md`) and will be connecting it as time allows. Eric is the **sole** contributor today.

## The core idea (trademark surprise as an innovation signal)

Trademark applications are subject to two competing pressures: applicants want claim language **wide** enough to cover their planned use, and **narrow** enough to actually be granted. That tension makes the application text a forced disclosure of what a new firm in a space imagined its business would be.

By measuring the **information content** (surprise) of each trademark - both prospectively (against priors available at filing time) and retrospectively (against what the industry then went on to do) - we can distinguish:

- Trademarks that were **obvious at the time** (low prospective surprise);
- Trademarks that were **only retrospectively obvious** (low retrospective surprise, high prospective);
- Trademarks that were **genuinely novel** (high on both axes).

The output of that decomposition is two industry-level series:

1. **Variation** within an industry at filing time - the diversity of guesses the field was making.
2. **Innovation** - the share (or weight) of filings that turned out, retrospectively, to have anticipated something real.

The data source is USPTO trademark filings. The publishable contribution is the construction of those series, plus whatever empirical claims they license about which industries innovate, when, and how.

## Connected research

This project does not stand alone. Eric is also working on:

- **O*NET-based simulation** of US-economy response to changing work eligibility and productivity shocks from automation. The simulation infrastructure shares modeling DNA with the trademark work and is the place where industry-level innovation signals would meaningfully feed back into a labor-market and consumption story.
- **Stone-Geary preferences** as the consumption-side anchor of the simulation. This formulation has dissertation-era pedigree in Eric's work and was proven to work; when extending the model, check whether the proved-to-work assertions are still load-bearing before adding new layers.

The framing: trademark surprise is a (new, underused) **measurement** layer; the O*NET/Stone-Geary work is the **simulation** layer; the paper opportunity is the join. AI coding tools have lowered the cost of running rich simulations enough that this combination is now tractable for a one-person research effort.

## Collaborators

**Eric is the sole contributor today.** No one else has worked on this project.

- **Seth Goldstein** (CMU, CS prof) - **invited; has not yet taken Eric up on it.** The invitation stands but cannot be assumed. Until Seth actively engages, do not write "we" in shared drafts, do not frame decisions as joint, and do not surface his personal context. If/when he joins, see `memory/person_seth_goldstein.md`.

**Not on this project:** Mark Kamlet was Eric's dissertation-era development partner historically (and the Stone-Geary lineage in Eric's broader work traces to that period), but he has not seen this trademark project and Eric does not expect him to. Do not list him as a collaborator, an advisor, or a likely interlocutor for this work.

The dissertation that this work descends from was a poor fit for CMU's behavioral research group, where Eric was housed. The structural reality of "fits badly where it lives" is part of the story of why it's resurfacing now rather than being already published.

## Target

A **modest paper** that complements existing literature. The framing matters - Eric does not want this inflated past what the data can support. The contribution being aimed at is:

- A clean operationalization of trademark surprise (the new measurement layer);
- A demonstration that the resulting series correlate sensibly with other innovation proxies (face validity);
- One or two empirical claims that are interesting *because they were hard to measure before*, not because they overturn anything.

Hold the bar there. Resist scope creep into "this rewrites the innovation literature" unless the data is screaming for it.

## What I expect Eric to pull in here

This folder is mostly empty by design - Eric has work on this project across the four-machine fleet that he'll be connecting and copying in. Likely categories:

- **Lit notes** - papers Eric has read on innovation measurement, USPTO data work, information-theoretic surprise / KL-divergence in economic data, and Stone-Geary applications.
- **Data inventory** - which USPTO bulk download(s) he is using, time coverage, what is cleaned vs raw, what derivations exist.
- **Code pointers** - links to the repos or working folders on the other machines (work laptop, MacBooks, CORE) that hold the simulation, the trademark cleaning, and the model. Keep these as **pointers**, not copies - shared writes from multiple machines will race.
- **Drafts** - any partial paper or memo text Eric has written, plus the framing material he might eventually send to Seth as part of the standing invitation.

When Eric pastes content in, prefer to **organize and index** it rather than restructure prose - he likely has more context for the wording than I do.

## Default operating posture (until Eric overrides)

- Treat Eric as a naif on the literature; deep-dive readings, name the schools, surface methodological debates.
- Hold the "modest paper" target as the bar; flag scope creep.
- When stress-testing claims (under `/critic`), probe the two highest-risk hinges: (a) whether trademark-text-derived surprise tracks the construct it is meant to track, and (b) whether the Stone-Geary assumptions remain load-bearing under the current extensions.
- The user is the only interlocutor on this project. Do not invoke "Seth would say..." or "Mark would push back on..." style framings; there is no joint thinking history to draw on here.

## Folder layout

```
trademark-driven-innovation-by-industry/
  PROJECT.md       this file - read first
  INBOX.md         running list of items Eric wants to pull in (next read)
  docs/            reference materials (papers, USPTO docs, attached PDFs)
  notes/           working notes, lit-review chunks, draft fragments
```
