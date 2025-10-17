# Judge-Aggregator

### Abstract

Please provide an **abstract-level summary** of your project **(max. 1000 characters)**.

Include the following:

* What is your primary research question?  
* What about your approach means you will provide clarity where existing work does not?  
* What preliminary experiments have you already conducted, and why do they indicate your approach is promising to pursue?

---

**Primary Research Question:** Which judges actually matter for human preferences, how do we systematically select them, and how robust are these patterns to different settings?

**Novel Approach:** While existing work evaluates LLM-as-a-judge systems, it often lacks principled aggregation methods and systematic interpretability analysis. Our approach combines three key innovations: (1) **Interpretability**: we use GAM-based aggregators to decompose which evaluation dimensions (judges) contribute most to human preference predictions, (2) **Automated Judge Selection**: we develop a pipeline to systematically identify and optimize judge panels for specific datasets, and (3) **Robustness Analysis**: we rigorously test aggregator behavior under realistic contamination scenarios (biased humans, biased judges). This provides clarity beyond existing work by making the aggregation process interpretable and by quantifying which evaluation dimensions matter most across different contexts.

**Preliminary Results:** Our workshop paper demonstrated promising initial results: learned aggregators (GAM R²=0.575, MLP R²=0.578) outperformed naive baselines by \~15% on synthetic persona data. GAM interpretability analysis revealed stable judge importance rankings across 20+ training runs, identifying Truthfulness, Instruction Following, and Clarity as consistently top contributors. Robustness experiments showed the aggregators maintained performance under moderate judge perturbations while revealing vulnerability to systematic training data contamination. These results validate that (a) learned aggregation improves over heuristics, (b) judge importance can be reliably extracted, and (c) the framework provides interpretable insights into preference modeling, forming a strong foundation for expanded research into judge selection automation, cross-task analysis, and real human data validation.

## Methodology

What is your approach, and how will you validate it?

* State your hypothesis and any novel techniques you’ll introduce  
* Describe experimental design, baselines, and method of comparison  
* Provide key assumptions your approach relies on  
* Outline explicitly how you’ll *confirm* or *deny* your hypothesis

---

Check out the “[Comprehensive Methodology](?tab=t.x8g9whnl2ni4)” tab for the complete methodology proposal. We're going to be working, mainly, on the first 4 tracks proposed.

### Initial validation

How would you validate your idea with a small-scale preliminary experiment? Fill out the [Initial Validation Worksheet](?tab=t.nym2pdz2agur), then summarize your planned experiment in one paragraph. 

---

### Highly relevant works

Create a list of highly relevant works. For each item, add a hyperlinked name and a 1-2 sentence TL;DR for *why it is relevant to your work*.

You will likely want to use the [Lit Review Template | Apart Fellowship](https://docs.google.com/document/d/17wWXzUgXAicgHuAbsLgeL2mSuUzNJVE4Ec9SimcnDyg/edit?usp=sharing) to help with this.

***REVIEWERS: If you have any suggestions that we have missed, please link them here\!***

---

\[Specific\] [LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks](https://aclanthology.org/2025.acl-short.20/)

TLDR: Explores at different granularities (based on expert/non-expert annotation and machine/human responses) for a diverse set of NLG datasets (70,000 rows), how LLM-as-a-Judge following annotator instructions correlates with annotator ratings.  They discover poor alignment for safety datasets as low as \-0.24 spearman correlation, while max average correlation looks like 0.5 Spearman and 0.28 Cohen’s k.

Relevance: High \- Useful baseline dataset, comparable Judge architecture being annotator instruction based (a naive way to align human preference). Only challenge \- uses Spearman correlation for understanding alignment

\[Eitan\]: Putting down some papers that seem relevant.

From Narmeen:

* [https://arxiv.org/html/2406.11657v1](https://arxiv.org/html/2406.11657v1)

Related to Datasets and Benchmark (some papers that did similar things to us, that I put as part of the Methodology section):

* [LLMs Instead of Human Judges](https://aclanthology.org/2025.acl-short.20.pdf)  
* [Multi-Agent as Judge](https://arxiv.org/pdf/2507.21028)  
* [Learning an Efficient Multi-turn Dialogue Evaluator from Multiple Judges](https://arxiv.org/pdf/2508.00454)

### People to talk to

Create a list of people to reach out to about the project; start with the authors of highly relevant works.

***REVIEWERS: If you have any suggestions or connections you can facilitate, please add/comment them here and we will coordinate with you to get an introduction\!***

---

### Proposed submissions

Where will you submit this work?

This applies to applications for funding, as well as workshop or conference submissions.

***REVIEWERS: If you know of any good opportunities that we missed, please link them here\!***

---

