import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from tqdm import tqdm
from datasets import load_dataset
from google.colab import drive

# drive.mount('/content/drive')

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to('cuda')

clean_translations_folder='/content/drive/MyDrive/clean_nllb200_distilled_600M/'

# NLLB does not use 2 character ISO code, so extra processing here
# the 7 character language codes are drawn from the NLLB 200 paper
lang_codes=['ar', 'ca', 'cy', 'de', 'et', 'id', 'fa', 'lv', 'mn', 'sl', 'sv', 'ta', 'tr', 'zh', 'ja']
# X + en -> en
csw_src = ['arb_Arab', 'cat_Latn', 'cym_Latn', 'deu_Latn', 'est_Latn', 'ind_Latn', 'pes_Arab', 'lvs_Latn', 'khk_Cyrl', 'slv_Latn', 'swe_Latn', 'tam_Taml', 'tur_Latn', 'zho_Hans', 'jpn_Jpan']
csw_tgt = 'eng_Latn'

# X + en -> X
reversecsw_src = 'eng_Latn'
reversecsw_tgt = ['arb_Arab', 'cat_Latn', 'cym_Latn', 'deu_Latn', 'est_Latn', 'ind_Latn', 'pes_Arab', 'lvs_Latn', 'khk_Cyrl', 'slv_Latn', 'swe_Latn', 'tam_Taml', 'tur_Latn', 'zho_Hans', 'jpn_Jpan']

# X -> en
monolingual_src = ['arb_Arab', 'cat_Latn', 'cym_Latn', 'deu_Latn', 'est_Latn', 'ind_Latn', 'pes_Arab', 'lvs_Latn', 'khk_Cyrl', 'slv_Latn', 'swe_Latn', 'tam_Taml', 'tur_Latn', 'zho_Hans', 'jpn_Jpan']
monolingual_tgt = 'eng_Latn'

# en -> X
reversemonolingual_src = 'eng_Latn'
reversemonolingual_tgt = ['arb_Arab', 'cat_Latn', 'cym_Latn', 'deu_Latn', 'est_Latn', 'ind_Latn', 'pes_Arab', 'lvs_Latn', 'khk_Cyrl', 'slv_Latn', 'swe_Latn', 'tam_Taml', 'tur_Latn', 'zho_Hans', 'jpn_Jpan']

import os

for lang_i, lang_code in enumerate(lang_codes):
    if lang_code not in ['ja', 'zh']: # these are scriptio continua languages that we're skipping
        continue
    print(f"Processing {lang_code}")

    dataset = load_dataset("sophiayk/CoVoSwitch", f"{lang_code}_en_combined", split='test')

    language_folder = clean_translations_folder+f"{lang_code}/"
    if not os.path.exists(language_folder):
        os.makedirs(language_folder)

    print("CSW processing...")
    with open(language_folder+"csw.txt", "w") as f_csw:
        for idx in tqdm(range(len(dataset))):

            # encoded_text = tokenizer(dataset[idx]['translation']['csw'], return_tensors="pt").to('cuda')
            # generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.get_lang_id("en"))
            # ans = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=csw_src[lang_i], tgt_lang=csw_tgt)
            output = translator(dataset[idx]['translation']['csw'])
            translated_text = output[0]['translation_text']

            #csw_data.append({"src": dataset[idx]['translation']['csw'], "mt": ans[0], "ref": dataset[idx]['translation']['en']})
            f_csw.write(translated_text + '\n')


    print("Reverse csw processing...")
    with open(language_folder+"reverse_csw.txt", "w") as f_reversecsw:
        for idx in tqdm(range(len(dataset))):
            # ca + en -> ca translation: reverse csw
            # tokenizer.src_lang = "en"
            # encoded_text = tokenizer(dataset[idx]['translation']['csw'], return_tensors="pt").to('cuda')
            # generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.get_lang_id(lang_code))
            # ans = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=reversecsw_src, tgt_lang=reversecsw_tgt[lang_i])
            output = translator(dataset[idx]['translation']['csw'])
            translated_text = output[0]['translation_text']

            #reverse_csw_data.append({"src": dataset[idx]['translation']['csw'], "mt": ans[0], "ref": dataset[idx]['translation'][lang_code]})
            f_reversecsw.write(translated_text + '\n')


    print("Monolingual processing...")
    with open(language_folder+"monolingual.txt", "w") as f_monolingual:
        for idx in tqdm(range(len(dataset))):
        # ca -> en translation: monolingual
            # if lang_code == 'sv-SE':
            #   tokenizer.src_lang = 'sv'
            # else:
            #   tokenizer.src_lang = lang_code
            # encoded_text = tokenizer(dataset[idx]['translation'][lang_code], return_tensors="pt").to('cuda')
            # generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.get_lang_id("en"))
            # ans = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=monolingual_src[lang_i], tgt_lang=monolingual_tgt)
            output = translator(dataset[idx]['translation'][lang_code])
            translated_text = output[0]['translation_text']

            #monolingual_data.append({"src": dataset[idx]['translation'][lang_code], "mt": ans[0], "ref": dataset[idx]['translation']['en']})
            f_monolingual.write(translated_text + '\n')

    print("Reverse monolingual processing...")
    with open(language_folder+"reverse_monolingual.txt", "w") as f_reversemonolingual:
        for idx in tqdm(range(len(dataset))):
            # en -> ca translation: reverse monolingual
            # tokenizer.src_lang = 'en'
            # encoded_text = tokenizer(dataset[idx]['translation']['en'], return_tensors="pt").to('cuda')
            # generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.get_lang_id(lang_code))
            # ans = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            translator=pipeline('translation', model=model, tokenizer=tokenizer, src_lang=reversemonolingual_src, tgt_lang=reversemonolingual_tgt[lang_i])
            output = translator(dataset[idx]['translation']['en'])
            translated_text = output[0]['translation_text']

            f_reversemonolingual.write(translated_text + '\n')
