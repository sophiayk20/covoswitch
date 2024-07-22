import evaluate
from datasets import load_dataset

lang_codes=['ar', 'ca', 'cy', 'de', 'et', 'fa', 'id', 'lv', 'mn', 'sl', 'sv', 'ta', 'tr']
for lang_code in lang_codes:
  with open(f'/content/drive/MyDrive/clean_m2m100_418M/{lang_code}/csw.txt', 'r') as f:
    csw = f.readlines()
    csw = list(map(lambda x: x.rstrip(), csw))

  with open(f'/content/drive/MyDrive/clean_m2m100_418M/{lang_code}/monolingual.txt', 'r') as f:
    monolingual = f.readlines()
    monolingual = list(map(lambda x: x.rstrip(), monolingual))

  with open(f'/content/drive/MyDrive/clean_m2m100_418M/{lang_code}/reverse_csw.txt', 'r') as f:
    reverse_csw = f.readlines()
    reverse_csw = list(map(lambda x: x.rstrip(), reverse_csw))

  with open(f'/content/drive/MyDrive/clean_m2m100_418M/{lang_code}/reverse_monolingual.txt', 'r') as f:
    reverse_monolingual = f.readlines()
    reverse_monolingual = list(map(lambda x: x.rstrip(), reverse_monolingual))

  print(f"Processing {lang_code}")
  metric = evaluate.load('sacrebleu')
  dataset = load_dataset('sophiayk/CoVoSwitch', f"{lang_code}_en_combined", split='test')
  #print(dataset[0])
  #print(dataset[1])

  references_en=[]
  references_X = []
  references_csw = []

  for i in range(len(dataset)):
    row = dataset[i]['translation']
    references_X.append([row[lang_code]])
    references_en.append([row['en']])
    references_csw.extend([row['csw']])

  #print(len(references_csw))
  #print(len(dataset))

  print("x + en -> en: " , f"{metric.compute(predictions=references_csw, references=references_en, tokenize='flores200')['score']:.1f}")
  print("x + en -> x: ", f"{metric.compute(predictions=references_csw, references=references_X, tokenize='flores200')['score']:.1f}")
  print("x -> en: ", f"{metric.compute(predictions=monolingual, references=references_en, tokenize='flores200')['score']:.1f}")
  print("en -> x: ", f"{metric.compute(predictions=reverse_monolingual, references=references_X, tokenize='flores200')['score']:.1f}")