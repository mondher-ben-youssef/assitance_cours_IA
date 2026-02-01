# 📚 Assistant IA de révision (RAG)
### LangChain + LangGraph + Streamlit + ChromaDB

---

## 🌐 Application en ligne

**L'application est déployée et accessible gratuitement :**  
👉 **[https://ia-cours-assistant.streamlit.app](https://ia-cours-assistant.streamlit.app)**

Vous pouvez l'utiliser directement sans installation locale !

---

## 📖 Description

**Assistant IA de révision** est une application intelligente de type **RAG (Retrieval Augmented Generation)** qui permet de **discuter avec vos documents PDF de cours**.

### 🎯 Fonctionnalités principales

✅ **Indexation intelligente de documents**
- Upload de fichiers PDF (cours, notes, supports pédagogiques)
- Découpage automatique en chunks pertinents
- Génération d'embeddings vectoriels
- Stockage dans une base vectorielle ChromaDB persistante

✅ **Recherche sémantique avancée**
- Récupération des passages les plus pertinents par rapport à votre question
- Paramètre `k` ajustable pour contrôler le nombre de passages récupérés

✅ **Génération de réponses contextuelles**
- Utilisation d'un LLM puissant (Groq API) pour générer des réponses précises
- Réponses basées **uniquement** sur le contenu de vos documents
- Ajout automatique des sources (document + numéro de page)

✅ **Interface intuitive**
- Interface web moderne développée avec Streamlit
- Historique de conversation
- Paramètres personnalisables (modèle LLM, nombre de passages)

---

## 🚀 Comment utiliser l'application

### Étape 1 : Uploader vos PDFs

1. Cliquez sur **"Ajoute tes PDFs"** dans la section **"1) Upload des PDFs"**
2. Sélectionnez un ou plusieurs fichiers PDF depuis votre ordinateur
3. Les formats acceptés : `.pdf`

### Étape 2 : Indexer les documents

1. Une fois vos PDFs uploadés, cliquez sur le bouton **🔎 Indexer**
2. L'application va :
   - Lire le contenu des PDFs
   - Découper le texte en chunks (segments cohérents)
   - Générer des embeddings vectoriels
   - Les stocker dans ChromaDB
3. Un message de confirmation apparaîtra : `Indexation terminée ✅ (X chunks ajoutés)`

### Étape 3 : Poser vos questions

1. Dans la section **"2) Poser des questions"**, tapez votre question
2. Exemples de questions :
   - *"Explique la différence entre processus et thread"*
   - *"Quels sont les algorithmes de tri et leurs complexités ?"*
   - *"Qu'est-ce que le Machine Learning supervisé ?"*
3. Cliquez sur **➡️ Répondre**
4. L'assistant va :
   - Rechercher les passages pertinents dans vos documents
   - Générer une réponse basée sur ces passages
   - Afficher les sources utilisées

### Étape 4 : Consulter l'historique

- L'historique des questions/réponses s'affiche en bas de page
- Permet de suivre votre session de révision

---

## ⚙️ Paramètres disponibles

### Dans la barre latérale :

- **Modèle Groq** : Choisir le modèle LLM (par défaut : `llama-3.1-8b-instant`)
- **k (passages récupérés)** : Nombre de passages à récupérer (2-8, défaut: 4)
- **🧹 Réinitialiser la base** : Supprimer tous les documents indexés et repartir à zéro

---

## 🛠️ Stack technique

| Composant | Technologie |
|-----------|-------------|
| **Interface utilisateur** | Streamlit |
| **🔗 Framework RAG** | **LangChain** |
| **🔀 Orchestration** | **LangGraph** (workflow retrieve → generate) |
| **Base vectorielle** | ChromaDB (persistée localement) |
| **Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **LLM** | Groq API (`llama-3.1-8b-instant`) |
| **Extraction PDF** | PyPDF |

---

## 🔗 Pourquoi LangChain ?

**LangChain** est le framework de référence pour construire des applications LLM. Dans ce projet, il est utilisé pour :

### 🎯 Gestion des embeddings et du retrieval
- **Intégration ChromaDB** : LangChain fournit une abstraction élégante pour interagir avec ChromaDB via `langchain-chroma`
- **Embeddings uniformes** : Utilisation de `sentence-transformers` via l'API standardisée de LangChain
- **Retrievers configurables** : Système de retrieval modulaire avec paramètre `k` ajustable

### 📝 Text Splitters intelligents
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Découpage intelligent qui préserve le contexte
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
```

### 🤖 Abstraction des LLMs
- **Multi-providers** : Facile de basculer entre Groq, OpenAI, Anthropic...
- **API unifiée** : Même interface quel que soit le provider
- **Gestion des prompts** : SystemMessage, HumanMessage, AIMessage standardisés

### 📚 Gestion des documents
- **Document loaders** : Extraction PDF avec métadonnées (source, page)
- **Schéma Document** : Structure standardisée `{page_content, metadata}`
- **Chaînes composables** : Pipeline retrieval → formatting → generation

---

## 🔀 Pourquoi LangGraph ?

**LangGraph** est le framework de nouvelle génération de LangChain pour orchestrer des workflows complexes. Il apporte :

### 🎭 Architecture à base de graphes

Notre application utilise un **workflow en 2 nœuds** :

```
┌─────────┐       ┌──────────┐       ┌─────┐
│  START  │  -->  │ RETRIEVE │  -->  │ GEN │  -->  END
└─────────┘       └──────────┘       └─────┘
```

#### Node 1 : **RETRIEVE**
```python
def retrieve_node(state: RAGState) -> RAGState:
    retriever = state["retriever"]
    question = state["question"]
    docs = retriever.get_relevant_documents(question)
    return {"docs": docs}
```
- Récupère les documents pertinents via le retriever LangChain
- Met à jour l'état du graphe avec les documents trouvés

#### Node 2 : **GENERATE**
```python
def generate_node(state: RAGState) -> RAGState:
    llm = make_llm(state["model_name"])
    context = _format_context(state["docs"])
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"CONTEXTE:\n{context}\n\nQUESTION: {state['question']}")
    ]
    
    response = llm.invoke(messages)
    return {"answer": response.content}
```
- Formate le contexte des documents récupérés
- Construit le prompt avec système + contexte + question
- Invoque le LLM et retourne la réponse

### ✨ Avantages de LangGraph

#### 🔄 État partagé typé
```python
class RAGState(TypedDict):
    messages: List[BaseMessage]  # Historique
    question: str                # Question courante
    docs: List[Document]         # Docs récupérés
    answer: str                  # Réponse finale
    retriever: object            # Retriever
    k: int                       # Nombre de passages
    model_name: str              # Modèle LLM
```
- État fortement typé évitant les erreurs
- Partagé entre tous les nœuds du graphe
- Immutable et traceable

#### 🎯 Modularité et extensibilité
- **Ajout facile de nœuds** : Ex. un nœud de re-ranking, de validation, de cache
- **Conditional edges** : Routage dynamique selon l'état
- **Parallel execution** : Possibilité d'exécuter plusieurs nœuds en parallèle

#### 🔍 Observabilité
- Chaque transition est tracée
- Debug facile du workflow
- Intégration avec LangSmith pour le monitoring

#### 🚀 Évolutions possibles

```
                    ┌──────────────┐
                    │   RETRIEVE   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
        ┌──────────►│   RE-RANK    │
        │           └──────┬───────┘
        │                  │
        │           ┌──────▼───────┐
        │           │   GENERATE   │
        │           └──────┬───────┘
        │                  │
        │           ┌──────▼───────┐
        └───────────┤   VALIDATE   │
         (retry)    └──────────────┘
```

### 🆚 Comparaison avec LCEL (LangChain Expression Language)

| Aspect | LCEL | LangGraph |
|--------|------|-----------|
| **Simplicité** | Chaînes simples | Workflows complexes |
| **État** | Passage de variables | État partagé global |
| **Conditionnalité** | Limitée | Routes conditionnelles |
| **Cycles** | Impossible | Support natif |
| **Debugging** | Difficile | Inspection d'état |
| **Use case** | RAG basique | RAG avancé, agents |

---

## 🏗️ Architecture technique détaillée

### Pipeline complet (indexation)

```
┌─────────────┐
│   PDF(s)    │
└──────┬──────┘
       │ PyPDF
       ▼
┌─────────────┐
│  Documents  │ (LangChain Document schema)
└──────┬──────┘
       │ RecursiveCharacterTextSplitter
       ▼
┌─────────────┐
│   Chunks    │ (1000 chars, overlap 200)
└──────┬──────┘
       │ Sentence-Transformers
       ▼
┌─────────────┐
│  Embeddings │ (384-dim vectors)
└──────┬──────┘
       │ langchain-chroma
       ▼
┌─────────────┐
│  ChromaDB   │ (persisted)
└─────────────┘
```

### Pipeline complet (question-réponse)

```
┌──────────────┐
│   Question   │
└──────┬───────┘
       │
       ▼
┌─────────────────────────┐
│   LangGraph Workflow    │
│  ┌─────────────────┐    │
│  │  RETRIEVE Node  │    │  ← Retriever LangChain
│  └────────┬────────┘    │  ← ChromaDB similarity search
│           │             │  ← Top k documents
│           ▼             │
│  ┌─────────────────┐    │
│  │  GENERATE Node  │    │  ← Format context
│  └────────┬────────┘    │  ← Build prompt (System + Context + Question)
│           │             │  ← ChatGroq LLM
└───────────┼─────────────┘
            │
            ▼
┌─────────────────┐
│  Réponse + Src  │
└─────────────────┘
```


## 💻 Installation locale

Si vous souhaitez exécuter l'application en local :

### Prérequis
- Python 3.9 ou supérieur
- Clé API Groq (gratuite sur [console.groq.com](https://console.groq.com))

### Instructions

1. **Cloner le projet**
```bash
git clone <url-du-repo>
cd assitance_cours_IA
```

2. **Créer un environnement virtuel**
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Configuration**

Créez un fichier `.env` à la racine du projet :
```env
GROQ_API_KEY=votre_clé_api_groq
GROQ_MODEL=llama-3.1-8b-instant
CHROMA_DIR=data/chroma
```

5. **Lancer l'application**
```bash
streamlit run app.py
```

L'application sera accessible sur `http://localhost:8501`

---

## 📁 Structure du projet

```
assitance_cours_IA/
│
├── app.py                  # Interface Streamlit principale
├── rag_pipeline.py         # Pipeline RAG (indexation, retrieval)
├── rag_graph.py            # Graph LangGraph (workflow retrieve → generate)
├── requirements.txt        # Dépendances Python
├── Dockerfile              # Configuration Docker
├── README.md               # Ce fichier
│
└── data/
    └── chroma/             # Base vectorielle ChromaDB (persistée)
```

---

## 🧪 Exemple d'utilisation

**Question :** *"Qu'est-ce que le gradient descent en Machine Learning ?"*

**Réponse générée :**
> Le gradient descent (descente de gradient) est un algorithme d'optimisation utilisé pour minimiser une fonction de coût. Il calcule le gradient (dérivée) de la fonction par rapport aux paramètres et met à jour ces paramètres dans la direction opposée au gradient...
>
> **Sources :**
> - ML_cours.pdf - page 12
> - ML_cours.pdf - page 13

---

## 🔐 Sécurité et confidentialité

- Les documents uploadés sont traités localement ou dans votre session Streamlit Cloud
- Les données ne sont pas partagées avec des tiers
- La clé API Groq est stockée de manière sécurisée (variables d'environnement)

---

## 👨‍💻 Auteur

Mondher Ben Youssef
Mehdi Jegham
Nabil Ghazouani 
Jouhayna Cheikh
Selim Khelifa