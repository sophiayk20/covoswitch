"""
    Make sure to have comet installed if you'd like to evaluate COMET this way.
    `pip install -q unbabel-comet==2.0.0` 

    This notebook evaluates COMET scores of raw code-switched inputs (baseline 1) and monolingual translations (baseline 2), then calculates deltas that code-switching system translations achieve from these two baselines.
"""
import numpy as np
from datasets import load_dataset
from comet import download_model, load_from_checkpoint

# load COMET model
model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)
comet_model.to('cuda')
comet_model.device # check that your comet model is on 'cuda'

lang_codes=['ar', 'ca', 'cy', 'de', 'et', 'fa', 'id', 'lv', 'mn', 'sl', 'sv', 'ta', 'tr']

# Calculate Baseline Scores
## Raw Code-Switched Inputs
for lang_code in lang_codes:
  print(f"Processing {lang_code}")

  dataset = load_dataset('sophiayk/CoVoSwitch', f"{lang_code}_en_combined", split='test')

  with open(f"/content/drive/MyDrive/clean_nllb200_distilled_600M/{lang_code}/csw.txt", "r") as f:
    nllb_translations_csw = f.readlines()
    nllb_translations_csw = list(map(lambda x: x.rstrip(), nllb_translations_csw))

  with open(f"/content/drive/MyDrive/clean_nllb200_distilled_600M/{lang_code}/reverse_csw.txt", "r") as f:
    nllb_translations_rcsw = f.readlines()
    nllb_translations_rcsw = list(map(lambda x: x.rstrip(), nllb_translations_rcsw))

  with open(f"/content/drive/MyDrive/clean_m2m100_418M/{lang_code}/csw.txt", "r") as f:
    m2m_translations_csw = f.readlines()
    m2m_translations_csw = list(map(lambda x: x.rstrip(), m2m_translations_csw))

  with open(f"/content/drive/MyDrive/clean_m2m100_418M/{lang_code}/reverse_csw.txt", "r") as f:
    m2m_translations_rcsw = f.readlines()
    m2m_translations_rcsw = list(map(lambda x: x.rstrip(), m2m_translations_rcsw))

  csw_data = []
  rcsw_data = []


  references_en=[]
  references_X = []
  references_csw = []

  for i in range(len(dataset)):
    row = dataset[i]['translation']
    references_en.append(row['en'])
    references_X.append(row[lang_code])
    references_csw.append(row['csw'])

  # Baseline
  print("Baselines")
  for ref_csw, ref_en in zip(references_csw, references_en):
    csw_data.append({"src": ref_csw, "mt": ref_csw, "ref": ref_en})

  for ref_rcsw, ref_X in zip(references_csw, references_X):
    rcsw_data.append({"src": ref_rcsw, "mt": ref_rcsw, "ref": ref_X})

  model_output = comet_model.predict(csw_data, batch_size=8, gpus=1)[0] # fetches list of scores
  baseline_csw_score = round(np.mean(model_output)*100, 1)
  print(f"CSW COMET Score: {baseline_csw_score}")

  model_output = comet_model.predict(rcsw_data, batch_size=8, gpus=1)[0] # fetches list of scores
  baseline_rcsw_score = round(np.mean(model_output)*100, 1)
  print(f"RCSW COMET Score: {baseline_rcsw_score}")

  # M2M100

  print("M2M100")

  m2m_csw_data = []
  m2m_rcsw_data = []

  # X + en -> en
  for ref_csw, trans_csw, ref_en in zip(ref_csw, m2m_translations_csw, references_en):
    m2m_csw_data.append({"src": ref_csw, "mt": trans_csw, "ref": ref_en})

  # X + en -> X
  for csw, trans_rcsw, ref_X in zip(ref_csw, m2m_translations_rcsw, references_X):
    m2m_rcsw_data.append({"src": ref_csw, "mt": trans_rcsw, "ref": ref_X})

  model_output = comet_model.predict(m2m_csw_data, batch_size=8, gpus=1)[0] # fetches list of scores
  score = round(np.mean(model_output)*100, 1)
  print(f"M2M CSW COMET Delta: {round(score-baseline_csw_score,1)}")

  model_output = comet_model.predict(m2m_rcsw_data, batch_size=8, gpus=1)[0] # fetches list of scores
  score = round(np.mean(model_output)*100, 1)
  print(f"M2M RCSW COMET Delta: {round(score-baseline_rcsw_score,1)}")

  # NLLB200
  print("NLLB200")

  nllb_csw_data = []
  nllb_rcsw_data = []

  # X + en -> en
  for ref_csw, trans_csw, ref_en in zip(ref_csw, nllb_translations_csw, references_en):
    nllb_csw_data.append({"src": ref_csw, "mt": trans_csw, "ref": ref_en})

  # X + en -> X
  for csw, trans_rcsw, ref_X in zip(ref_csw, nllb_translations_rcsw, references_X):
    nllb_rcsw_data.append({"src": ref_csw, "mt": trans_rcsw, "ref": ref_X})

  model_output = comet_model.predict(nllb_csw_data, batch_size=8, gpus=1)[0] # fetches list of scores
  score = round(np.mean(model_output)*100, 1)
  print(f"NLLB CSW COMET Delta: {round(score-baseline_csw_score,1)}")

  model_output = comet_model.predict(nllb_rcsw_data, batch_size=8, gpus=1)[0] # fetches list of scores
  score = round(np.mean(model_output)*100, 1)
  print(f"NLLB RCSW COMET Delta: {round(score-baseline_rcsw_score,1)}")


## Monolingual Baselines
  for lang_code in lang_codes:
    print(f"Processing {lang_code}")

    dataset = load_dataset('sophiayk/CoVoSwitch', f"{lang_code}_en_combined", split='test')

    with open(f"/content/drive/MyDrive/clean_nllb200_distilled_600M/{lang_code}/monolingual.txt", "r") as f:
        nllb_trans_mono = f.readlines()
        nllb_trans_mono= list(map(lambda x: x.rstrip(), nllb_trans_mono))

    with open(f"/content/drive/MyDrive/clean_nllb200_distilled_600M/{lang_code}/reverse_monolingual.txt", "r") as f:
        nllb_trans_rmono = f.readlines()
        nllb_trans_rmono = list(map(lambda x: x.rstrip(), nllb_trans_rmono))

    with open(f"/content/drive/MyDrive/clean_m2m100_418M/{lang_code}/monolingual.txt", "r") as f:
        m2m_trans_mono = f.readlines()
        m2m_trans_mono = list(map(lambda x: x.rstrip(), m2m_trans_mono))

    with open(f"/content/drive/MyDrive/clean_m2m100_418M/{lang_code}/reverse_monolingual.txt", "r") as f:
        m2m_trans_rmono = f.readlines()
        m2m_trans_rmono = list(map(lambda x: x.rstrip(), m2m_trans_rmono))

    csw_data = []
    rcsw_data = []

    references_en=[]
    references_X = []
    references_csw = []

    for i in range(len(dataset)):
        row = dataset[i]['translation']
        references_en.append(row['en'])
        references_X.append(row[lang_code])
        references_csw.append(row['csw'])

    # Baseline
    print("Baselines")
    for ref_csw, ref_en in zip(references_csw, references_en):
        csw_data.append({"src": ref_csw, "mt": ref_csw, "ref": ref_en})

    for ref_rcsw, ref_X in zip(references_csw, references_X):
        rcsw_data.append({"src": ref_rcsw, "mt": ref_rcsw, "ref": ref_X})

    model_output = comet_model.predict(csw_data, batch_size=8, gpus=1)[0] # fetches list of scores
    baseline_csw_score = round(np.mean(model_output)*100, 1)
    print(f"CSW COMET Baseline Score: {baseline_csw_score}")

    model_output = comet_model.predict(rcsw_data, batch_size=8, gpus=1)[0] # fetches list of scores
    baseline_rcsw_score = round(np.mean(model_output)*100, 1)
    print(f"RCSW COMET Baseline Score: {baseline_rcsw_score}")

    # M2M100

    print("M2M100")

    m2m_csw_data = []
    m2m_rcsw_data = []

    # X + en -> en
    for ref_csw, trans_mono, ref_en in zip(ref_csw, m2m_trans_mono, references_en):
        m2m_csw_data.append({"src": ref_csw, "mt": trans_mono, "ref": ref_en})

    # X + en -> X
    for csw, trans_rmono, ref_X in zip(ref_csw, m2m_trans_rmono, references_X):
        m2m_rcsw_data.append({"src": ref_csw, "mt": trans_rmono, "ref": ref_X})

    model_output = comet_model.predict(m2m_csw_data, batch_size=8, gpus=1)[0] # fetches list of scores
    score = round(np.mean(model_output)*100, 1)
    print(f"M2M CSW COMET X + en -> en score: {round(score,1)}")

    model_output = comet_model.predict(m2m_rcsw_data, batch_size=8, gpus=1)[0] # fetches list of scores
    score = round(np.mean(model_output)*100, 1)
    print(f"M2M RCSW COMET X + en -> X score: {round(score, 1)}")

    # NLLB200
    print("NLLB200")

    nllb_csw_data = []
    nllb_rcsw_data = []

    # X + en -> en
    for ref_csw, trans_mono, ref_en in zip(ref_csw, nllb_trans_mono, references_en):
        nllb_csw_data.append({"src": ref_csw, "mt": trans_mono, "ref": ref_en})

    # X + en -> X
    for csw, trans_rmono, ref_X in zip(ref_csw, nllb_trans_rmono, references_X):
        nllb_rcsw_data.append({"src": ref_csw, "mt": trans_rmono, "ref": ref_X})

    model_output = comet_model.predict(nllb_csw_data, batch_size=8, gpus=1)[0] # fetches list of scores
    score = round(np.mean(model_output)*100, 1)
    print(f"NLLB CSW COMET X + en -> en: {round(score, 1)}")

    model_output = comet_model.predict(nllb_rcsw_data, batch_size=8, gpus=1)[0] # fetches list of scores
    score = round(np.mean(model_output)*100, 1)
    print(f"NLLB RCSW COMET X + en -> X: {round(score, 1)}")

## Comet Deltas, Relative to Monolingual Baselines
for lang_code in lang_codes[::-1]:
  print(f"Processing {lang_code}")

  dataset = load_dataset('sophiayk/CoVoSwitch', f"{lang_code}_en_combined", split='test')

  # FETCH MONOLINGUAL
  with open(f"/content/drive/MyDrive/clean_nllb200_distilled_600M/{lang_code}/monolingual.txt", "r") as f:
    nllb_translations_mono = f.readlines()
    nllb_translations_mono = list(map(lambda x: x.rstrip(), nllb_translations_mono))

  with open(f"/content/drive/MyDrive/clean_nllb200_distilled_600M/{lang_code}/reverse_monolingual.txt", "r") as f:
    nllb_translations_rmono = f.readlines()
    nllb_translations_rmono = list(map(lambda x: x.rstrip(), nllb_translations_rmono))

  with open(f"/content/drive/MyDrive/clean_m2m100_418M/{lang_code}/monolingual.txt", "r") as f:
    m2m_translations_mono = f.readlines()
    m2m_translations_mono = list(map(lambda x: x.rstrip(), m2m_translations_mono))

  with open(f"/content/drive/MyDrive/clean_m2m100_418M/{lang_code}/reverse_monolingual.txt", "r") as f:
    m2m_translations_rmono = f.readlines()
    m2m_translations_rmono = list(map(lambda x: x.rstrip(), m2m_translations_rmono))

  # FETCH CODE-SWITCHING
  with open(f"/content/drive/MyDrive/clean_nllb200_distilled_600M/{lang_code}/csw.txt", "r") as f:
    nllb_translations_csw = f.readlines()
    nllb_translations_csw = list(map(lambda x: x.rstrip(), nllb_translations_csw))

  with open(f"/content/drive/MyDrive/clean_nllb200_distilled_600M/{lang_code}/reverse_csw.txt", "r") as f:
    nllb_translations_rcsw = f.readlines()
    nllb_translations_rcsw = list(map(lambda x: x.rstrip(), nllb_translations_rcsw))

  with open(f"/content/drive/MyDrive/clean_m2m100_418M/{lang_code}/csw.txt", "r") as f:
    m2m_translations_csw = f.readlines()
    m2m_translations_csw = list(map(lambda x: x.rstrip(), m2m_translations_csw))

  with open(f"/content/drive/MyDrive/clean_m2m100_418M/{lang_code}/reverse_csw.txt", "r") as f:
    m2m_translations_rcsw = f.readlines()
    m2m_translations_rcsw = list(map(lambda x: x.rstrip(), m2m_translations_rcsw))

  nllb_mono_data = []
  nllb_rmono_data = []

  m2m_mono_data = []
  m2m_rmono_data = []

  references_en=[]
  references_X = []
  references_csw = []

  for i in range(len(dataset)):
    row = dataset[i]['translation']
    references_en.append(row['en'])
    references_X.append(row[lang_code])
    references_csw.append(row['csw'])

  # Baseline
  print("Baselines")
  # M2M X -> en
  for ref_x, trans_mono, ref_en in zip(references_X, m2m_translations_mono, references_en):
    m2m_mono_data.append({"src": ref_x, "mt": trans_mono, "ref": ref_en})

  # M2M en -> X
  for ref_en, trans_rmono, ref_X in zip(references_en, m2m_translations_rmono, references_X):
    m2m_rmono_data.append({"src": ref_en, "mt": trans_rmono, "ref": ref_X})

  # NLLB X -> en
  for ref_x, trans_mono, ref_en in zip(references_X, nllb_translations_mono, references_en):
    nllb_mono_data.append({"src": ref_x, "mt": trans_mono, "ref": ref_en})

  # NLLB en -> X
  for ref_en, trans_rmono, ref_X in zip(references_en, nllb_translations_rmono, references_X):
    nllb_rmono_data.append({"src": ref_en, "mt": trans_rmono, "ref": ref_X})


  print("M2M")

  model_output = comet_model.predict(m2m_mono_data, batch_size=8, gpus=1)[0] # fetches list of scores
  m2m_baseline_mono_score = round(np.mean(model_output)*100, 1)
  print(f"BASELINE MONO COMET Score: {m2m_baseline_mono_score}")

  model_output = comet_model.predict(m2m_rmono_data, batch_size=8, gpus=1)[0] # fetches list of scores
  m2m_baseline_rmono_score = round(np.mean(model_output)*100, 1)
  print(f"BASELINE RMONO COMET Score: {m2m_baseline_rmono_score}")


  print("NLLB")

  model_output = comet_model.predict(nllb_mono_data, batch_size=8, gpus=1)[0] # fetches list of scores
  nllb_baseline_mono_score = round(np.mean(model_output)*100, 1)
  print(f"BASELINE MONO COMET Score: {nllb_baseline_mono_score}")

  model_output = comet_model.predict(nllb_rmono_data, batch_size=8, gpus=1)[0] # fetches list of scores
  nllb_baseline_rmono_score = round(np.mean(model_output)*100, 1)
  print(f"BASELINE RMONO COMET Score: {nllb_baseline_rmono_score}")

  # M2M100

  print("M2M100")

  m2m_csw_data = []
  m2m_rcsw_data = []

  # X + en -> en
  for ref_csw, trans_csw, ref_en in zip(references_csw, m2m_translations_csw, references_en):
    m2m_csw_data.append({"src": ref_csw, "mt": trans_csw, "ref": ref_en})

  # X + en -> X
  for ref_csw, trans_rcsw, ref_X in zip(references_csw, m2m_translations_rcsw, references_X):
    m2m_rcsw_data.append({"src": ref_csw, "mt": trans_rcsw, "ref": ref_X})

  model_output = comet_model.predict(m2m_csw_data, batch_size=8, gpus=1)[0] # fetches list of scores
  score = round(np.mean(model_output)*100, 1)
  print(f"M2M CSW COMET Delta: {round(score-m2m_baseline_mono_score,1)}")

  model_output = comet_model.predict(m2m_rcsw_data, batch_size=8, gpus=1)[0] # fetches list of scores
  score = round(np.mean(model_output)*100, 1)
  print(f"M2M RCSW COMET Delta: {round(score-m2m_baseline_rmono_score,1)}")

  # NLLB200
  print("NLLB200")

  nllb_csw_data = []
  nllb_rcsw_data = []

  # X + en -> en
  for ref_csw, trans_csw, ref_en in zip(references_csw, nllb_translations_csw, references_en):
    nllb_csw_data.append({"src": ref_csw, "mt": trans_csw, "ref": ref_en})

  # X + en -> X
  for ref_csw, trans_rcsw, ref_X in zip(references_csw, nllb_translations_rcsw, references_X):
    nllb_rcsw_data.append({"src": ref_csw, "mt": trans_rcsw, "ref": ref_X})

  model_output = comet_model.predict(nllb_csw_data, batch_size=8, gpus=1)[0] # fetches list of scores
  score = round(np.mean(model_output)*100, 1)
  print(f"NLLB CSW COMET Delta: {round(score-nllb_baseline_mono_score,1)}")

  model_output = comet_model.predict(nllb_rcsw_data, batch_size=8, gpus=1)[0] # fetches list of scores
  score = round(np.mean(model_output)*100, 1)
  print(f"NLLB RCSW COMET Delta: {round(score-nllb_baseline_rmono_score,1)}")

## Comet Deltas Relative to Raw Code-Switched Input Baselines
for lang_code in lang_codes:
  print(f"Processing {lang_code}")

  dataset = load_dataset('sophiayk/CoVoSwitch', f"{lang_code}_en_combined", split='test')

  # FETCH MONOLINGUAL
  with open(f"/content/drive/MyDrive/clean_nllb200_distilled_600M/{lang_code}/monolingual.txt", "r") as f:
    nllb_translations_mono = f.readlines()
    nllb_translations_mono = list(map(lambda x: x.rstrip(), nllb_translations_mono))

  with open(f"/content/drive/MyDrive/clean_nllb200_distilled_600M/{lang_code}/reverse_monolingual.txt", "r") as f:
    nllb_translations_rmono = f.readlines()
    nllb_translations_rmono = list(map(lambda x: x.rstrip(), nllb_translations_rmono))

  with open(f"/content/drive/MyDrive/clean_m2m100_418M/{lang_code}/monolingual.txt", "r") as f:
    m2m_translations_mono = f.readlines()
    m2m_translations_mono = list(map(lambda x: x.rstrip(), m2m_translations_mono))

  with open(f"/content/drive/MyDrive/clean_m2m100_418M/{lang_code}/reverse_monolingual.txt", "r") as f:
    m2m_translations_rmono = f.readlines()
    m2m_translations_rmono = list(map(lambda x: x.rstrip(), m2m_translations_rmono))

  # FETCH CODE-SWITCHING
  with open(f"/content/drive/MyDrive/clean_nllb200_distilled_600M/{lang_code}/csw.txt", "r") as f:
    nllb_translations_csw = f.readlines()
    nllb_translations_csw = list(map(lambda x: x.rstrip(), nllb_translations_csw))

  with open(f"/content/drive/MyDrive/clean_nllb200_distilled_600M/{lang_code}/reverse_csw.txt", "r") as f:
    nllb_translations_rcsw = f.readlines()
    nllb_translations_rcsw = list(map(lambda x: x.rstrip(), nllb_translations_rcsw))

  with open(f"/content/drive/MyDrive/clean_m2m100_418M/{lang_code}/csw.txt", "r") as f:
    m2m_translations_csw = f.readlines()
    m2m_translations_csw = list(map(lambda x: x.rstrip(), m2m_translations_csw))

  with open(f"/content/drive/MyDrive/clean_m2m100_418M/{lang_code}/reverse_csw.txt", "r") as f:
    m2m_translations_rcsw = f.readlines()
    m2m_translations_rcsw = list(map(lambda x: x.rstrip(), m2m_translations_rcsw))

  nllb_mono_data = []
  nllb_rmono_data = []

  m2m_mono_data = []
  m2m_rmono_data = []

  csw_data = []
  rcsw_data = []

  references_en=[]
  references_X = []
  references_csw = []

  for i in range(len(dataset)):
    row = dataset[i]['translation']
    references_en.append(row['en'])
    references_X.append(row[lang_code])
    references_csw.append(row['csw'])

  # Baseline
  print("Baselines")
  # CSW, en
  for ref_csw, ref_en in zip(references_csw, references_en):
    csw_data.append({"src": ref_csw, "mt": ref_csw, "ref": ref_en})

  # CSW, X
  for ref_csw, ref_X in zip(references_csw, references_X):
    rcsw_data.append({"src": ref_csw, "mt": ref_csw, "ref": ref_X})

  print("M2M")

  model_output = comet_model.predict(csw_data, batch_size=8, gpus=1)[0]
  baseline_csw_score = round(np.mean(model_output)*100, 1)
  print(f"Baseline CSW COMET Score: {baseline_csw_score}")

  model_output = comet_model.predict(rcsw_data, batch_size=8, gpus=1)[0]
  baseline_rcsw_score = round(np.mean(model_output)*100, 1)
  print(f"Baseline RCSW COMET Score: {baseline_rcsw_score}")

  #M2M100
  print("M2M100")

  m2m_csw_data = []
  m2m_rcsw_data = []

  # X + en -> en
  for ref_csw, trans_csw, ref_en in zip(references_csw, m2m_translations_csw, references_en):
    m2m_csw_data.append({"src": ref_csw, "mt": trans_csw, "ref": ref_en})

  # X + en -> X
  for ref_csw, trans_rcsw, ref_X in zip(references_csw, m2m_translations_rcsw, references_X):
    m2m_rcsw_data.append({"src": ref_csw, "mt": trans_rcsw, "ref": ref_X})

  model_output = comet_model.predict(m2m_csw_data, batch_size=8, gpus=1)[0] # fetches list of scores
  score = round(np.mean(model_output)*100, 1)
  print(f"M2M CSW COMET Delta: {round(score-baseline_csw_score,1)}")

  model_output = comet_model.predict(m2m_rcsw_data, batch_size=8, gpus=1)[0] # fetches list of scores
  score = round(np.mean(model_output)*100, 1)
  print(f"M2M RCSW COMET Delta: {round(score-baseline_rcsw_score,1)}")

  # NLLB200
  print("NLLB200")

  nllb_csw_data = []
  nllb_rcsw_data = []

  # X + en -> en
  for ref_csw, trans_csw, ref_en in zip(references_csw, nllb_translations_csw, references_en):
    nllb_csw_data.append({"src": ref_csw, "mt": trans_csw, "ref": ref_en})

  # X + en -> X
  for ref_csw, trans_rcsw, ref_X in zip(references_csw, nllb_translations_rcsw, references_X):
    nllb_rcsw_data.append({"src": ref_csw, "mt": trans_rcsw, "ref": ref_X})

  model_output = comet_model.predict(nllb_csw_data, batch_size=8, gpus=1)[0] # fetches list of scores
  score = round(np.mean(model_output)*100, 1)
  print(f"NLLB CSW COMET Delta: {round(score-baseline_csw_score,1)}")

  model_output = comet_model.predict(nllb_rcsw_data, batch_size=8, gpus=1)[0] # fetches list of scores
  score = round(np.mean(model_output)*100, 1)
  print(f"NLLB RCSW COMET Delta: {round(score-baseline_rcsw_score,1)}")
