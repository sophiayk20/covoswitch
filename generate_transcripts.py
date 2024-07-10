import os
import argparse
from tqdm import tqdm
from datasets import load_dataset

def generate():
    for lang_code in lang_codes:
        dataset = load_dataset('covost2', lang_code, cache_dir="./cache", data_dir=".")
        ds = dataset[CV_SUBSET]

        source, target = lang_code.split('_')

        print(f"lang code {lang_code} >>> ")

        langcodedir = RAWTEXT_DIR+f'/{lang_code}' 

        if not os.path.exists(langcodedir):
            os.makedirs(langcodedir)
            
        with open(f"{langcodedir}/en.txt", "w") as f_en:
            with open(f"{langcodedir}/{target}.txt", "w") as f_tgt:
                with open(f"{langcodedir}/{lang_code}-usedfiles.txt", "w") as f_cv:
                    for idx, instance in enumerate(tqdm(ds)):
                        sentence = instance['sentence'].replace('"', '')
                        translation = instance['translation'].replace('"', '')

                        # English transcription
                        f_en.write(sentence + '\n')

                        # Target transcription
                        f_tgt.write(translation + '\n')

                        # Common Voice filename used
                        f_cv.write(instance['file'] + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate transcriptions')
    parser.add_argument("--subset", type=str, choices=['train', 'validation', 'test'], help="Common Voice subset, one of 'train', 'validation', 'test'")

    args = parser.parse_args()

    # these are the lang codes for CoVoST 2
    lang_codes = ["en_de", "en_tr", "en_fa", "en_sv-SE", "en_mn", "en_cy", "en_ca", "en_sl", "en_et",
                "en_id", "en_ar", "en_ta", "en_lv"]

    # Choose from 'train', 'validation', and 'test' as appropriate
    CV_SUBSET=args.subset
    # path to where transcripts will be created
    RAWTEXT_DIR = f'./{CV_SUBSET}-ds'

    if not os.path.exists(RAWTEXT_DIR):
        os.makedirs(RAWTEXT_DIR)

    generate()

