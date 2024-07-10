# CoVoSwitch: Machine Translation of Synthetic Code-Switched Text Based on Intonation Units

## Paper & Dataset
This repository holds code used in writing "CoVoSwitch: Machine Translation of Synthetic Code-Switched Text Based on Intonation Units," to be published in *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics: Student Research Workshop*. Paper was accepted on July 9, 2024 for ACL-SRW 2024 (Aug. 11-16 2024).

The dataset I used in this paper is available through HuggingFace. It was created using CoVoST 2, which is a speech-to-text translation dataset created in turn from Common Voice Corpus 4.0. 

Link to paper: [to be made available here].
Dataset: https://huggingface.co/sophiayk/covoswitch.

If you'd like to use this code or paper, please cite:
```
    Yeeun Kang. 2024. CoVoSwitch: Machine Translation of Synthetic Code-Switched Text Based on Intonation Units. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics: Student Research Workshop, pages -, Bangkok, Thailand. Association for Computational Linguistics. 
```

## Pre-existing theories, datasets, models, and metrics used in this paper
- **Theories**: Matrix-Language Embedded Framework (1997), Intonation Unit Boundary Constraint (2023)
- **Speech-to-Text Translation Dataset**: CoVoST 2 (2021)
- **Intonation Unit Detection**: PSST (2023), based on Whisper (2023)
- **Word-level text-to-text aligner**: awesome-align (2021)
- **Multilingual Neural Machine Translation (MNMT) Models**: Meta's M2M-100 418M (2021), NLLB-200 600M (2022)
- **Metrics**: chrF++ (2017, character level), spBLEU (2022, tokenized language-agnostic metric level), COMET (2022, models human judgments of translations).

## Code Structure
The code structure is divided into subsections of *Section 2: Synthetic Data Generation* described in the paper.
- **Intonation Unit Detection**:
- **Alignment Generation**:
- **Intonation Unit Replacement**:
- **Dataset Evaluation and Analysis**:

All code were run on Google Colab with a single NVIDIA L4 GPU.

## Paper Summary
There are 2 main code-switching linguistic theories I discovered while writing this paper. 
- **Matrix-Language Embedded Framework (MLF)**: keeps one language as the matrix language and includes segments from an embedded language. 
- **Equivalence Constraint (EC) Theory**: grammatical structure present in one language must also exist in the other - as such, parsers are needed for code-switching text creation.

One difficulty that comes with using the Equivalence Constraint is that it requires the use of constituency parsers. I found two parsers I could potentially use - the Stanford Parser and the Berkeley Neural Parser, but these only support a specific subset of languages. I realized that I could not use this approach to create code-switched text in unstudied languages, but found that the speech domain had datasets covering low-resource languages I wanted to study.

So instead I paid attention to a recent prosodic constraint established in EMNLP 2023. 
- **Intonation Unit Boundary Constraint**: code-switching is more common across intonation units than within as a result of looser syntactic relationships.

Meanwhile, I also found that OpenAI's STT model Whisper can be fine-tuned to detect English prosodic boundaries [the PSST model, published in CoNLL 2023](https://aclanthology.org/2023.conll-1.31/). This means that Whisper can segment an English utterance into intonation units. 

Following the **Matrix-Language Embedded Framework** and the **Intonation Unit Boundary Constraint**, I synthesized code-switching text data by switching languages using intonation units as basis units.

## Tech Stack
Python, HuggingFace transformers & datasets, Shell scripts

## Future Directions
Before I graduate in May 2025, I plan to create a fleursswitch dataset, created from Google's FLEURS. This will also be made available through my HuggingFace profile. I am also experimenting with non-English language pairs, which I am partially presenting at Interspeech YFRSW 2024 this year (Kos, Greece), but this is mainly based on the speech-to-speech modality.

## Paper Review
Feedback I received from ACL reviewers on this paper are publicly available on OpenReview.

## Other Tools of Potential Interest
To those who find my repository or paper interesting, here are some further repositories that I discovered while writing this paper that may be of interest.
- GCM: [GCM](https://github.com/microsoft/CodeMixed-Text-Generator), which generates code-switched text based on the EC Theory. Benepar & Stanford are both available as parsers and they offer a web interface for one-time generation.

## P.S. Personal Motivation & Background
I grew up bilingual, splitting my time in the US & Korea, and my sister and I always converse with one another in a perfect mix of both languages. We have never been able to explain why we use some words in English and others in Korean, but recent research [EMNLP Proceedings 2023](https://aclanthology.org/2023.emnlp-main.1047/) demonstrated that code-switching is more common across intonation units, which I found very interesting. In the future, written and spoken language systems will likely need to handle multilingual inputs, and I believe code-switching research is an interesting starting point to advancing towards them. I tested MNMT models with my synthetic dataset because I had always been interested in multimodal translation systems, but I see many other potential fields of application.

This project was done independently outside of college & work during my gap year from Yale (Apr. 2024 - Jun. 2024).
I'm happy to discuss the project further in any capacity through the email on my GitHub profile!