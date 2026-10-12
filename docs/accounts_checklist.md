# Accounts checklist (all free, no credit card)

Do these yourself; never paste tokens in chat or commit them.

- [ ] **GitHub**: create the public repo `road-damage-tn`, push this skeleton, check that CI turns green.
- [ ] **Kaggle**: account + phone verification (needed to enable the free GPU). Settings > API > create a token (only needed for downloading datasets via CLI).
- [ ] **Hugging Face**: account, then later a Dataset repo (Tunisian data) and a Space (demo). Create a *write* access token when we publish.
- [ ] **DagsHub** (optional): only if you want DVC remote storage there; otherwise we use Hugging Face Datasets.
- [ ] **Roboflow** (optional): not required, we annotate with CVAT or Label Studio.
- [ ] **Google Colab**: nothing to create if you have a Google account (backup GPU).

Before relying on any free-tier limit (Kaggle GPU hours, HF storage), check the provider's current page.

## Decisions to record (defaults proposed, awaiting your confirmation)
- Dataset license: CC BY 4.0
- First regions: Ariana / Grand Tunis, then Sfax and a rural or highway trip
- Data versioning: Hugging Face Datasets (+ DVC/DagsHub optional)
- Second annotator: someone from ASIA or your class
