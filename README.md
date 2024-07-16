# CoVoSwitch: Machine Translation of Synthetic Code-Switched Text Based on Intonation Units

## Paper & Dataset
This repository holds code used in writing "CoVoSwitch: Machine Translation of Synthetic Code-Switched Text Based on Intonation Units," to be published in *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics: Student Research Workshop*. Paper was accepted on July 9, 2024 for ACL-SRW 2024 (Aug. 11-16 2024).

CoVoSwitch, the dataset I created in this paper, is available through HuggingFace. It was created using CoVoST 2, which is a speech-to-text translation dataset created in turn from Common Voice Corpus 4.0. 

Link to paper: [to be made available here].

Dataset: https://huggingface.co/sophiayk/covoswitch.

If you'd like to use this code or paper, please cite:
```
    Yeeun Kang. 2024. CoVoSwitch: Machine Translation of Synthetic Code-Switched Text Based on Intonation Units. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics: Student Research Workshop, pages -, Bangkok, Thailand. Association for Computational Linguistics. 
```
## Topics & Keywords
Multilingual code-switching translation (NMT), prosodic speech segmentation (intonation unit detection) using Whisper (STT)

Keywords: code-switching, intonation units, machine translation, low-resource languages, speech-to-text, multilingual

## Theories, datasets, models, and metrics used in this paper
- **Theories**: Matrix-Language Embedded Framework (1997), Intonation Unit Boundary Constraint (2023)
- **Speech-to-Text Translation Dataset**: CoVoST 2 (2021, specifically the En:rightarrow:X subset), based on Common Voice Corpus 4.0 (2020)
- **Intonation Unit Detection**: PSST (2023), based on Whisper (2023)
- **Word-level text-to-text aligner**: awesome-align (2021)
- **Multilingual Neural Machine Translation (MNMT) Models**: Meta's M2M-100 418M (2021), NLLB-200 600M (2022)
- **Metrics**: chrF++ (2017, character level), spBLEU (2022, tokenized language-agnostic metric level), COMET (2022, models human judgments of translations).

## Code Structure & Generation Pipeline
The code structure mostly matches the subsections of *Section 2: Synthetic Data Generation* described in the paper.
- **Download Common Voice 4.0**: [Download here](https://commonvoice.mozilla.org/en/datasets) into your local workspace.
- **Process CoVoST 2**: **generate_transcripts.py**
    - Since CoVoST 2 is a speech-to-text translation dataset, we need to process both speech and text in the dataset.
    - Usage: `python generate_transcripts.py --subset train`
        - --subset argument must either be `train`, `validation`, or `test`
    - Creates a directory with 3 files per language pair:
        - 1. English transcriptions (text): will be used to check whether PSST generates correct IU boundary-marked transcripts (used to check against transcription error).
        - 2. Target translations (text): will be used in alignment step
        - 3. Common Voice filenames used (text): will be used as input to PSST for IU boundary-marked transcripts.
- *Intermediate step: Move Common Voice files to Google Drive (skip this step if you're using GPU locally)*
    - `shell-scripts/tarzip_valid.sh`: is an example of how to zip local files. You can subsequently upload them into your remote workspace (like Google Drive for Colab).
- **Intonation Unit Detection**: **detect_intonation_unit.py**
    - With English Common Voice files, we generate English transcripts marked with intonation unit boundaries by PSST.
    - Usage: `python detect_IU.py --englishfilename "some English file path" --boundaryfilename "some boundary file path"`.
    - Creates intonation unit boundary-marked English transcriptions.
    - PSST performs two functionalities, details of which can be found in Table 1 of the paper.
        - Transcribing
        - marking IU boundaries.
    - `--englishfilename`: feed name of text file that stores all English Common Voice 4.0 filenames.
    - `--boundaryfilename`: feed name of text file that will store all IU-detected transcripts, each separated by a newline.
- **Alignment Generation**:
- **Intonation Unit Replacement**:
- **Dataset Evaluation and Analysis**:
- **Running and Evaluating Translation Models**:

All code (except for Common Voice transcript generation) were run on Google Colab with a single NVIDIA L4 GPU. Jupyter notebooks have been converted to Python scripts to facilitate future use. I have also included shell scripts I used in this project in the directory `shell-scripts`. `tarzip_valid.sh` is a shell script that I used to transfer Common Voice audio files from local to Google Drive after zipping them into tar. 

## Tech Stack
Python, HuggingFace transformers & datasets, Shell scripts

HuggingFace transformers - AutoTokenizer, AutoProcessor, AutoModel, AutoModelForSpeechSeq2Seq

Audio libraries: librosa

## Paper Summary
There are 2 main code-switching linguistic theories I discovered while writing this paper. 
- **Matrix-Language Embedded Framework (MLF)**: keeps one language as the matrix language and includes segments from an embedded language. 
- **Equivalence Constraint (EC) Theory**: grammatical structure present in one language must also exist in the other - as such, parsers are needed for code-switching text creation.

One difficulty that comes with using the Equivalence Constraint is that it requires the use of constituency parsers. I found two parsers I could potentially use - the Stanford Parser and the Berkeley Neural Parser, but these only support a specific subset of languages. I realized that I could not use this approach to create code-switched text in unstudied languages, but found that the speech domain had datasets covering low-resource languages I wanted to study.

So instead I paid attention to a recent prosodic constraint established in EMNLP 2023. 
- **Intonation Unit Boundary Constraint**: code-switching is more common across intonation units than within as a result of looser syntactic relationships.

Meanwhile, I also found that OpenAI's STT model Whisper can be fine-tuned to detect English prosodic boundaries [the PSST model, published in CoNLL 2023](https://aclanthology.org/2023.conll-1.31/). This means that Whisper can segment an English utterance into intonation units. 

Following the **Matrix-Language Embedded Framework** and the **Intonation Unit Boundary Constraint**, I synthesized code-switching text data by switching languages using intonation units as basis units.

## Future Directions
Before I graduate in May 2025, I plan to create a fleursswitch dataset, created from Google's FLEURS. This will also be made available through my HuggingFace profile. I am also experimenting with non-English language pairs, which I am partially presenting at *Interspeech YFRSW 2024* this year (Kos, Greece), but this is mainly based on the speech-to-speech modality. 

I note that this project was created with an English intonation unit detection model. I'm excited to see future work for IU detection on non-English languages.

## Paper Review
Feedback I received from ACL reviewers on this paper are publicly available on OpenReview.

## Other Tools of Potential Interest
To those who find my repository or paper interesting, here are some further repositories that I discovered while writing this paper that may be of interest.
- GCM: [GCM](https://github.com/microsoft/CodeMixed-Text-Generator), which generates code-switched text based on the EC Theory. Benepar & Stanford are both available as parsers and they offer a web interface for one-time generation.

## P.S. Personal Motivation & Background
I grew up bilingual, splitting my time in the US & Korea, and my sister and I always converse with one another in a perfect mix of both languages. We have never been able to explain why we use some words in English and others in Korean, but recent research [EMNLP Proceedings 2023](https://aclanthology.org/2023.emnlp-main.1047/) demonstrated that code-switching is more common across intonation units, which I found very interesting. In the future, written and spoken language systems will likely need to handle multilingual inputs, and I believe code-switching research is an interesting starting point to advancing towards them. I tested MNMT models with my synthetic dataset because I had always been interested in multimodal translation systems, but I see many other potential fields of application.

This project was done independently outside of college & work during my gap year from Yale (Apr. 2024 - Jun. 2024).
I'm happy to discuss the project further in any capacity through the email on my GitHub profile!