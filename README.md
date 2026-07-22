# BETO Fine-tuned para Detección de Discurso de Odio (HatEval-ES)

Fine-tuning de [BETO](https://github.com/dccuchile/beto) (`bert-base-spanish-wwm-cased`) para la detección de discurso de odio en español, evaluando si el ajuste supervisado reorganiza el espacio de embeddings de forma más útil que el modelo preentrenado sin ajustar. Proyecto final de la materia de Deep Learning — Universidad San Francisco de Quito (USFQ).

## Resumen

Se entrena BETO con *mean pooling* sobre el subconjunto en español de [HatEval](https://huggingface.co/datasets/valeriobasile/HatEval) (SemEval-2019 Task 5, 6,599 tuits), evaluando un *ablation study* de congelamiento progresivo de capas (0/25/50/75/100%). Los embeddings resultantes se comparan contra BETO sin ajustar mediante validación cruzada, evaluación en un conjunto de prueba independiente, y análisis de separabilidad geométrica (t-SNE, *silhouette score*).

## Resultados principales (semilla 43)

| Métrica | BETO puro | BETO ajustado |
|---|---|---|
| F1 (CV, mejor clasificador — SVM) | 0.735 | 0.945 |
| Accuracy (test) | — | 0.830 (IC 95%: [0.807, 0.854]) |
| F1 macro (test) | — | 0.83 |
| Silhouette score (test) | 0.025 | 0.294 |

El modelo ajustado supera el mejor resultado del *shared task* original de 2019 (F1 macro = 0.730, mineriaUNAM/Atalaya/MITRE) en +0.10 puntos absolutos (+14% relativo). Una verificación con una semilla aleatoria distinta (44) confirmó que, si bien la configuración óptima exacta de congelamiento no es estable entre ejecuciones, la magnitud de la mejora del *fine-tuning* frente a BETO puro sí lo es.

## Contenido del repositorio

```
.
├── hateval_beto_finetuning_final.ipynb   # Notebook completo: datos, ablation, entrenamiento, evaluación
├── paper_ieee.tex                        # Artículo científico (formato IEEE)
├── paper_ieee.pdf                        # Artículo compilado
├── IEEEtran.cls                          # Clase LaTeX requerida para compilar
├── fig_*.png                             # Figuras del paper (curvas, t-SNE)
└── README.md
```

## Cómo correrlo

El notebook está pensado para ejecutarse en Google Colab (requiere GPU):

**[Abrir en Google Colab](https://colab.research.google.com/drive/10bv8kWDUymPMRMlF4uqEl5N8HQT1szIe?usp=sharing)**

Antes de correr, es necesario:
1. Aceptar los términos de acceso del dataset [HatEval en Hugging Face](https://huggingface.co/datasets/valeriobasile/HatEval) con tu propia cuenta.
2. Configurar un token de Hugging Face (`HF_TOKEN`) como Secret de Colab.

## Metodología

- **Modelo base**: `dccuchile/bert-base-spanish-wwm-cased` (BETO).
- **Pooling**: promedio ponderado por `attention_mask` sobre `last_hidden_state` (*mean pooling*), no `pooler_output`.
- **Ablation study**: congelamiento progresivo de capas (0/25/50/75/100%), motivado por la razón entre parámetros entrenables y ejemplos de entrenamiento (~23,600 parámetros/ejemplo con el modelo completo descongelado).
- **Clasificadores clásicos evaluados sobre los embeddings**: Regresión Logística, SVM (RBF), Random Forest, Gradient Boosting.
- **Validación**: partición 70/15/15 fija y reutilizada en todas las etapas para evitar fuga de información; validación cruzada estratificada repetida (5×5) sobre `train+val`; evaluación final única sobre `test`.

## Limitaciones

- El dataset fue recolectado en 2018–2019; el lenguaje de odio en redes sociales evoluciona.
- El alcance se limita a discurso de odio hacia dos objetivos (mujeres e inmigrantes).
- La configuración óptima exacta del *ablation* no es estable entre semillas aleatorias (ver verificación de robustez en el paper).

## Referencias

- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT.
- Cañete et al. (2020). *Spanish Pre-Trained BERT Model and Evaluation Data*. PML4DC at ICLR.
- Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP-IJCNLP.
- Basile et al. (2019). *SemEval-2019 Task 5: Multilingual Detection of Hate Speech Against Immigrants and Women in Twitter*. SemEval-2019.
- Jawahar et al. (2019). *What Does BERT Learn about the Structure of Language?*. ACL.

## Autor

Fernando José Soto Jácome — Universidad San Francisco de Quito (USFQ)

## Licencia

MIT License — ver [LICENSE](LICENSE).
