import os
from datasets import load_dataset
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm


model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to('cuda')
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

clean_translations_folder='/content/drive/MyDrive/clean_m2m100_418M/'

lang_codes=['ar', 'ca', 'cy', 'de', 'et', 'id', 'fa', 'lv', 'mn', 'sl', 'sv', 'ta', 'tr']

for lang_code in lang_codes:
  print(f"Processing {lang_code}")

  dataset = load_dataset("sophiayk/CoVoSwitch", f"{lang_code}_en_combined", split='test')

  language_folder = clean_translations_folder+f"{lang_code}/"
  if not os.path.exists(language_folder):
    os.makedirs(language_folder)

  print("CSW processing...")
  with open(language_folder+"csw.txt", "w") as f_csw:
    for idx in tqdm(range(len(dataset))):
      encoded_text = tokenizer(dataset[idx]['translation']['csw'], return_tensors="pt").to('cuda')
      generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.get_lang_id("en"))
      ans = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
      #csw_data.append({"src": dataset[idx]['translation']['csw'], "mt": ans[0], "ref": dataset[idx]['translation']['en']})
      f_csw.write(ans[0] + '\n')

  print("Reverse csw processing...")
  with open(language_folder+"reverse_csw.txt", "w") as f_reversecsw:
    for idx in tqdm(range(len(dataset))):
      # ca + en -> ca translation: reverse csw
      tokenizer.src_lang = "en"
      encoded_text = tokenizer(dataset[idx]['translation']['csw'], return_tensors="pt").to('cuda')
      generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.get_lang_id(lang_code))
      ans = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
      #reverse_csw_data.append({"src": dataset[idx]['translation']['csw'], "mt": ans[0], "ref": dataset[idx]['translation'][lang_code]})
      f_reversecsw.write(ans[0] + '\n')


  print("Monolingual processing...")
  with open(language_folder+"monolingual.txt", "w") as f_monolingual:
    for idx in tqdm(range(len(dataset))):
    # ca -> en translation: monolingual
      if lang_code == 'sv-SE':
        tokenizer.src_lang = 'sv'
      else:
        tokenizer.src_lang = lang_code
      encoded_text = tokenizer(dataset[idx]['translation'][lang_code], return_tensors="pt").to('cuda')
      generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.get_lang_id("en"))
      ans = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
      #monolingual_data.append({"src": dataset[idx]['translation'][lang_code], "mt": ans[0], "ref": dataset[idx]['translation']['en']})
      f_monolingual.write(ans[0] + '\n')

  print("Reverse monolingual processing...")
  with open(language_folder+"reverse_monolingual.txt", "w") as f_reversemonolingual:
    for idx in tqdm(range(len(dataset))):
      # en -> ca translation: reverse monolingual
      tokenizer.src_lang = 'en'
      encoded_text = tokenizer(dataset[idx]['translation']['en'], return_tensors="pt").to('cuda')
      generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.get_lang_id(lang_code))
      ans = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
      f_reversemonolingual.write(ans[0] + '\n')
