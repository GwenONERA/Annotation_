# Walkthrough — Pipeline d'Analyse d'Erreurs EMOTYC

## Résumé

Pipeline en 3 scripts Python séquentiels pour évaluer, diagnostiquer et comprendre les erreurs du modèle EMOTYC (CamemBERT fine-tuné pour 11 émotions) via annotation par Claude Sonnet 4.6 (AWS Bedrock).

## Scripts créés

### 1. [emotyc_predict.py](file:///home/gwen/annotation/scripts/emotyc_predict.py) — Inférence EMOTYC locale

- Charge le modèle depuis HuggingFace (`TextToKids/CamemBERT-base-EmoTextToKids`)
- Template `before:</s>current:{s}</s>after:</s>` (bca_v3, meilleur template identifié)
- Swap admiration ↔ autre (identifié dans le notebook retroIngenierie)
- **Seuils optimisés par défaut** (issus du corpus de 2451 phrases), désactivables avec `--no-optimized-thresholds`
- Option `--use-context` pour inclure les phrases voisines (i-1, i+1)
- Exporte `emotyc_predictions.jsonl` (avec probas, prédictions binaires, gold labels, divergences)

```bash
python scripts/emotyc_predict.py \
    --xlsx outputs/homophobie/annotations_validees.xlsx \
    --out_dir outputs/homophobie/emotyc_eval
```

---

### 2. [emotyc_llm_judge.py](file:///home/gwen/annotation/scripts/emotyc_llm_judge.py) — Annotation LLM

Deux passes sur les lignes divergentes uniquement :

**Passe double-blind** : Le LLM juge entre "Annotateur A/B" (randomisé avec seed fixe) sans connaître l'identité (machine vs humain). Anti-sycophancy by design.

**Passe diagnostic** : Le LLM reçoit les probas EMOTYC + gold labels et classifie chaque erreur selon :
- Taxonomie d'erreurs : `lexical_argot`, `ironie_polarity`, `pragmatic_confusion`, `seuil_limite`, `erreur_humain`, `contexte_manquant`, `autre`
- Axes pragmatiques : ressentie / provoquée / thématisée / absente

```bash
python scripts/emotyc_llm_judge.py \
    --predictions outputs/homophobie/emotyc_eval/emotyc_predictions.jsonl \
    --out_dir outputs/homophobie/emotyc_eval
```

---

### 3. [emotyc_report.py](file:///home/gwen/annotation/scripts/emotyc_report.py) — Rapport statistique

Agrège tous les résultats :
- Métriques classiques (F1, κ, accuracy par émotion)
- Distribution des types d'erreurs
- Cohérence du juge LLM (double-blind vs diagnostic)
- Analyse croisée émotion × type d'erreur
- Export CSV + figures PNG avec `--export`

```bash
python scripts/emotyc_report.py \
    --eval_dir outputs/homophobie/emotyc_eval --export
```

## Modifications aux fichiers existants

```diff:requirements.txt
# Core
boto3
pandas
openpyxl

# Comparaison / visualisation
scikit-learn
matplotlib
seaborn
numpy

# Supervision manuelle (notebook)
ipywidgets

# HuggingFace provider
openai
===
# Core
boto3
pandas
openpyxl

# Comparaison / visualisation
scikit-learn
matplotlib
seaborn
numpy

# Supervision manuelle (notebook)
ipywidgets

# HuggingFace provider
openai

# EMOTYC local inference
torch
transformers
```

## Vérification

| Test | Résultat |
|---|---|
| `py_compile` (3 scripts) | ✓ Tous passent |
| `--help` emotyc_llm_judge.py | ✓ Fonctionne |
| `--help` emotyc_report.py | ✓ Fonctionne |
| `--help` emotyc_predict.py | ⚠ Échoue car `torch` non installé localement (attendu) |

> [!NOTE]
> [emotyc_predict.py](file:///home/gwen/annotation/scripts/emotyc_predict.py) nécessite un GPU avec `torch` et `transformers`. Il a été vérifié syntaxiquement via `py_compile` mais ne peut pas être testé fonctionnellement dans cet environnement.

## Design anti-biais

- **Anti-sycophancy** : La passe double-blind randomise l'attribution A/B avec un seed fixe. Le LLM ne sait pas qui est le gold, qui est EMOTYC.
- **Argot** : Les deux prompts demandent explicitement l'analyse de l'argot 11-25 ans.
- **Pragmatique** : Distinction ressentie/provoquée/thématisée pour identifier les confusions d'EMOTYC.
- **Résumé** : Le LLM doit produire un résumé court dans `justification_stricte` / `justification` pour garder les outputs concis.
