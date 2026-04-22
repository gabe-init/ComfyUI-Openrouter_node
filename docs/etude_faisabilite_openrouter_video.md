# Etude de faisabilite - extension video du node ComfyUI OpenRouter

Date de reference: 2026-04-23

## Objectif

Etudier la faisabilite d'etendre le node `ComfyUI-Openrouter_node` pour:

- exposer davantage de modeles OpenRouter dans ComfyUI
- inclure les modeles video OpenRouter
- preparer une architecture propre pour des modeles comme `alibaba/wan-*`, `bytedance/seedance-*`, `kwaivgi/kling-video-o1`, `openai/sora-2-pro` et `google/veo-3.1`

Cette etude a ete faite sans lancer de generation video payante. Elle s'appuie sur:

- l'analyse du code local du depot
- la documentation officielle OpenRouter
- les endpoints publics de catalogue de modeles OpenRouter

## Resume executif

Oui, c'est faisable.

En revanche, ce n'est pas un simple ajout de modeles dans la liste existante. Le node actuel est construit autour de `POST /api/v1/chat/completions`, alors que la generation video OpenRouter passe par une API dediee et asynchrone:

- `POST /api/v1/videos`
- `GET /api/v1/videos/{jobId}`
- `GET /api/v1/videos/{jobId}/content`

Conclusion pratique:

- pour exposer les modeles video, il faut au minimum une nouvelle branche d'execution
- le plus propre est probablement de garder le node actuel pour le chat/multimodal texte-image-PDF, et d'ajouter un node video distinct
- une partie du travail consiste aussi a corriger le catalogue des modeles, car le node actuel ne charge pas tous les modeles OpenRouter

## Constat sur le depot actuel

### 1. Le node actuel est un node de chat, pas un node video

Dans [node.py](K:\Codex_Projects\ComfyUI_OpenRouter\node.py:175), la methode principale `generate_response(...)` envoie toujours les requetes vers:

- `https://openrouter.ai/api/v1/chat/completions`

Le parser de reponse ne traite que:

- du texte
- des images renvoyees dans `message.images`

La sortie du node est declaree ainsi dans [node.py](K:\Codex_Projects\ComfyUI_OpenRouter\node.py:95):

- `("STRING", "IMAGE", "STRING", "STRING")`

Il n'y a donc aujourd'hui:

- ni type de sortie `VIDEO`
- ni polling asynchrone
- ni telechargement d'un fichier video genere

### 2. Le chargement de modeles est incomplet

Dans [node.py](K:\Codex_Projects\ComfyUI_OpenRouter\node.py:102), `fetch_openrouter_models()` appelle:

- `https://openrouter.ai/api/v1/models`

Sans parametre `output_modalities=all`.

Or, d'apres la documentation OpenRouter, l'endpoint `/api/v1/models` retourne par defaut les modeles `text` seulement. Les modeles image-only, video, embeddings, rerank, etc. ne sont pas tous remontes dans cette vue par defaut.

Verification publique faite le 2026-04-22:

- `GET /api/v1/models` retourne `349` modeles
- `GET /api/v1/models?output_modalities=all` retourne `397` modeles

Les modeles manquants dans la liste actuelle se repartissent ainsi:

- `8` modeles `video`
- `10` modeles `image`
- `25` modeles `embeddings`
- `3` modeles `rerank`
- `2` modeles `speech`

Cela explique pourquoi certains modeles OpenRouter ne sont pas visibles dans le node actuel, meme s'ils existent bien cote OpenRouter.

### 3. Certains inputs UI sont presents mais pas exploites

Le node expose deja les widgets:

- `aspect_ratio`
- `image_resolution`
- `seed`

Mais dans la charge utile envoyee a OpenRouter, seuls `model`, `messages`, `temperature` et `seed` sont ajoutes explicitement dans [node.py](K:\Codex_Projects\ComfyUI_OpenRouter\node.py:321).

Concretement:

- `seed` est bien envoye
- `aspect_ratio` n'est pas branche au payload actuel
- `image_resolution` n'est pas branche non plus

Ce point est important, parce que les modeles video OpenRouter ont justement besoin d'un vrai mapping de:

- `resolution`
- `aspect_ratio`
- `duration`
- parfois `size`
- parfois des options provider-specifiques

## Ce que permet OpenRouter aujourd'hui

### 1. Distinction cruciale entre chat multimodal et generation video

OpenRouter expose au moins deux familles utiles pour nous:

1. `chat/completions`
   Pour texte, image, audio, PDF, et aussi analyse de video en entree via `video_url`.
2. `videos`
   Pour la generation video elle-meme.

Ce point est central: la generation video ne se fait pas via `chat/completions`.

### 2. Video en entree vs video en sortie

Il faut bien separer deux cas d'usage:

- **Video en entree**: envoyer une video a un modele qui sait l'analyser
- **Video en sortie**: demander la generation d'une video

OpenRouter documente les videos en entree via `chat/completions` avec un bloc `video_url`.

En revanche, la generation video se fait via l'API dediee `/api/v1/videos`, avec un workflow asynchrone.

### 3. L'API video est asynchrone

D'apres la doc OpenRouter:

- on soumet un job a `POST /api/v1/videos`
- on recupere immediatement un `id`, un `polling_url` et un statut
- on interroge ensuite `GET /api/v1/videos/{jobId}`
- quand le statut est `completed`, on telecharge le media

Statuts documentes:

- `pending`
- `in_progress`
- `completed`
- `failed`

### 4. Le catalogue video est disponible publiquement

Le 2026-04-22, l'endpoint public `GET https://openrouter.ai/api/v1/videos/models` a renvoye au moins les modeles suivants:

- `kwaivgi/kling-video-o1`
- `bytedance/seedance-2.0`
- `bytedance/seedance-2.0-fast`
- `alibaba/wan-2.7`
- `alibaba/wan-2.6`
- `bytedance/seedance-1-5-pro`
- `openai/sora-2-pro`
- `google/veo-3.1`

Donc, sur le fond, les modeles que tu veux viser sont bien exposes aujourd'hui par OpenRouter.

### 5. Les modeles video ont des capacites heterogenes

Le catalogue video OpenRouter retourne par modele des metadonnees utiles:

- resolutions supportees
- aspect ratios supportes
- durees supportees
- presence ou non d'audio
- parametres pass-through autorises

Exemple de besoins variables suivant les modeles:

- `generate_audio`
- `frame_images` pour image-to-video
- `input_references` pour reference-to-video
- `provider.options` pour certains reglages specifiques au fournisseur

Conclusion:

- on ne doit pas coder une UI video figee autour d'un seul modele
- il faut une couche de description des capacites par modele

### 6. Image-to-video et controle first/last frame

Ton besoin d'avoir une image en entree, voire un mode image-entree + image-sortie de controle pour la video, est coherent avec ce que publie OpenRouter aujourd'hui.

Verification publique faite le 2026-04-22 sur `GET /api/v1/videos/models`:

- `kwaivgi/kling-video-o1` expose `supported_frame_images = [first_frame, last_frame]`
- `bytedance/seedance-2.0` expose `supported_frame_images = [first_frame, last_frame]`
- `bytedance/seedance-2.0-fast` expose `supported_frame_images = [first_frame, last_frame]`
- `bytedance/seedance-1-5-pro` expose `supported_frame_images = [first_frame, last_frame]`
- `alibaba/wan-2.7` expose `supported_frame_images = [first_frame, last_frame]`
- `alibaba/wan-2.6` expose `supported_frame_images = [first_frame]`
- `openai/sora-2-pro` n'expose pas publiquement `supported_frame_images`
- `google/veo-3.1` n'expose pas publiquement `supported_frame_images`

Lecture pratique:

- **Kling, Seedance et Wan 2.7** sont de tres bons candidats pour un vrai mode image-to-video avec image initiale et eventuellement image finale
- **Wan 2.6** semble plutot adapte a un mode image-to-video avec image de depart seulement
- **Sora 2 Pro** et **Veo 3.1** peuvent rester visibles dans le node video, mais sans activer automatiquement les controles `first_frame` et `last_frame` tant que leurs capacites exactes n'ont pas ete confirmees dans les metadonnees ou via un test volontaire

Nuance importante:

- les descriptions publiees pour `openai/sora-2-pro` et `google/veo-3.1` parlent bien de generation a partir de texte ou d'image
- en revanche, le catalogue video public ne detaille pas pour eux un schema `supported_frame_images` comme pour Kling, Seedance ou Wan
- il faut donc les traiter comme compatibles image-conditionnee de maniere prudente, sans promettre d'emblee un vrai mode `first_frame` / `last_frame`

Autrement dit:

- oui, le mode "je donne une image de depart" est clairement faisable
- oui, le mode "je donne une image de depart et une image cible/finale" est aussi faisable pour plusieurs modeles
- cette fonctionnalite doit etre consideree comme un axe central du futur node video

## Faisabilite technique

### Verdict

Faisable, avec une faisabilite elevee si on accepte une architecture en deux nodes.

### Ce qui est facile

- corriger le catalogue de modeles pour inclure `output_modalities=all`
- recuperer le sous-catalogue video depuis `/api/v1/videos/models`
- exposer les modeles video dans une UI dediee
- mapper les images ComfyUI vers `frame_images` ou `input_references`
- implementer le polling OpenRouter

### Ce qui demande plus d'attention

- produire une vraie sortie `VIDEO` exploitable dans ComfyUI
- gerer proprement le telechargement et le cycle de vie du fichier video
- choisir comment presenter les differences entre modeles sans rendre le node illisible
- eviter toute execution involontaire vu le cout des generations

## Architecture recommandee

### Option recommandee: separer en plusieurs nodes

Je recommande:

1. **Conserver le node actuel** comme node chat/multimodal
   Nom cible possible: `OpenRouter Chat`
2. **Ajouter un node dedie video**
   Nom cible possible: `OpenRouter Video`
3. **Partager une couche commune de catalogue**
   Un helper commun pour:
   - `models?output_modalities=all`
   - `videos/models`
   - cache local
   - normalisation des metadonnees

Pourquoi cette approche est preferable:

- le contrat d'entree/sortie n'est pas le meme
- le cycle d'execution n'est pas le meme
- les parametres ne sont pas les memes
- le cout et les temps de reponse ne sont pas comparables

### Option alternative: un seul mega-node

Possible, mais je la recommande moins.

Inconvenients:

- interface plus confuse
- logique conditionnelle plus lourde
- plus de risques de regressions sur le chat existant
- plus difficile a maintenir

## Proposition de design pour le futur node video

### Inputs minimaux

- `api_key`
- `prompt`
- `model` (source: `/api/v1/videos/models`)
- `resolution`
- `aspect_ratio`
- `duration`
- `seed`
- `generate_audio`

### Inputs optionnels

- `image_1` pour `frame_images:first_frame`
- `image_2` pour `frame_images:last_frame`
- `reference_image_1..N` pour `input_references`
- `provider_options_json` pour les cas avances
- `save_prefix` ou `output_name`

### Outputs proposes

- `video`
- `status`
- `metadata_json`
- `credits` ou `cost`

Note importante:

Le type exact du retour `VIDEO` devra etre aligne avec la version de ComfyUI cible. La documentation ComfyUI montre bien l'existence de nodes natifs qui retournent `VIDEO`, donc la faisabilite d'integration est bonne, mais il faudra verifier le format exact attendu dans l'environnement d'installation vise.

## UX recommandee pour le mode image-to-video

Le node video devrait proposer un selecteur de mode clair, par exemple:

- `text_to_video`
- `image_to_video`
- `start_end_frame_to_video`
- `reference_to_video`

### Comportement recommande

- `text_to_video`
  Aucun input image obligatoire
- `image_to_video`
  Un `first_frame` obligatoire
- `start_end_frame_to_video`
  Un `first_frame` obligatoire et un `last_frame` optionnel ou obligatoire selon le modele
- `reference_to_video`
  Une ou plusieurs images de reference, sans les traiter comme des frames strictes

### Logique de garde recommandee

- si le modele ne publie pas `last_frame`, masquer ou desactiver l'input de frame finale
- si le modele ne publie que `first_frame`, limiter l'UI a l'image de depart
- si le modele choisi ne publie pas de capacite image claire, rester en `text_to_video` par defaut
- si l'utilisateur branche a la fois des `frame_images` et des `input_references`, suivre le comportement OpenRouter documente: `frame_images` prend la priorite

### Priorite produit

Si on doit prioriser, je recommande cet ordre:

1. `text_to_video`
2. `image_to_video` avec `first_frame`
3. `start_end_frame_to_video` avec `first_frame` + `last_frame`
4. `reference_to_video`

Ce serait le meilleur compromis entre valeur immediate et complexite.

## Mapping OpenRouter -> ComfyUI a prevoir

### Text-to-video

Payload OpenRouter cible:

```json
{
  "model": "openai/sora-2-pro",
  "prompt": "A cinematic drone shot over a futuristic city",
  "resolution": "720p",
  "aspect_ratio": "16:9",
  "duration": 8,
  "seed": 42
}
```

### Image-to-video

Payload OpenRouter cible:

```json
{
  "model": "alibaba/wan-2.7",
  "prompt": "The character slowly turns and smiles",
  "frame_images": [
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/png;base64,..."
      },
      "frame_type": "first_frame"
    }
  ],
  "resolution": "1080p"
}
```

### Start/end-frame-to-video

Payload OpenRouter cible:

```json
{
  "model": "kwaivgi/kling-video-o1",
  "prompt": "A smooth transformation from stillness to a confident runway walk",
  "frame_images": [
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/png;base64,..."
      },
      "frame_type": "first_frame"
    },
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/png;base64,..."
      },
      "frame_type": "last_frame"
    }
  ],
  "resolution": "720p",
  "aspect_ratio": "16:9",
  "duration": 5
}
```

### Reference-to-video

Payload OpenRouter cible:

```json
{
  "model": "google/veo-3.1",
  "prompt": "A luxury product shot with elegant motion",
  "input_references": [
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/png;base64,..."
      }
    }
  ],
  "aspect_ratio": "16:9"
}
```

## Risques et points de vigilance

### 1. Cout

La video coute cher. Il faut absolument eviter:

- les appels automatiques accidentels
- les re-executions inutiles
- les changements d'inputs qui relancent un job sans intention explicite

Recommendation forte:

- ajouter un mode "dry run" ou "payload preview"
- rendre l'execution video tres explicite
- ne jamais auto-tester en developpement sans validation humaine

### 2. Temps d'attente

La doc OpenRouter indique que la generation video peut prendre de quelques dizaines de secondes a plusieurs minutes. Le node devra gerer:

- polling raisonnable
- timeout configurable
- statut visible
- erreurs lisibles

### 3. Heterogeneite des modeles

Tous les modeles video n'acceptent pas:

- les memes resolutions
- les memes durees
- les memes aspect ratios
- l'audio
- les memes options provider

Il faut donc baser les menus dynamiques sur les metadonnees renvoyees par `/videos/models`, pas sur des listes codees en dur.

Point important pour le "sans fond" / "fond transparent":

- au 2026-04-23, le catalogue video public OpenRouter expose bien les resolutions, aspect ratios, tailles, durees et certains `allowed_passthrough_parameters`
- en revanche, il n'expose pas aujourd'hui de champ unifie et fiable du type `transparent_background` pour tous les modeles video
- conclusion pratique: il ne faut pas inventer un switch universel dans le node tant que cette capacite n'est pas publiee explicitement par modele ou documentee de maniere stable cote provider/OpenRouter
- la bonne approche est de preparer la decouverte des `allowed_passthrough_parameters` par modele, puis d'ajouter un controle dedie seulement pour les modeles qui publient vraiment cette capacite

### 4. Zero Data Retention

La doc OpenRouter precise que la generation video n'est pas compatible avec ZDR, car la sortie doit etre conservee temporairement pour etre recuperee apres le traitement asynchrone.

Il faudra le documenter clairement dans l'UI ou le README.

## Roadmap recommandee

### Phase 1 - sans depense de credits

- corriger le catalogue de modeles du node actuel
- ajouter une couche de catalogue partagee
- preparer un node video en lecture seule
- afficher les modeles video et leurs capacites
- cabler l'UI sans lancer de job

### Phase 2 - implementation video

- implementer `POST /api/v1/videos`
- implementer le polling
- telecharger le fichier final
- retourner un objet `VIDEO` ou un fallback compatible

Etat du depot apres cette etape de travail:

- un node `OpenRouter Video` a ete ajoute avec:
  - champ `api_key`
  - logique de soumission async, polling et telechargement
  - validation locale sur une partie des contraintes publiees par modele
  - erreurs de soumission plus detaillees en cas de rejet par OpenRouter
- les widgets `mode`, `duration`, `resolution` et `aspect_ratio` sont maintenant pilotes par les capacites publiees du modele selectionne
- la duree minimale publiee par OpenRouter est automatiquement re-selectionnee quand on change de modele
- le catalogue video public actuellement expose est integre dynamiquement, y compris `Wan 2.6`, `Wan 2.7`, `Seedance 1.5 Pro`, `Seedance 2.0`, `Seedance 2.0 Fast`, `Kling Video O1`, `Sora 2 Pro` et `Veo 3.1`
- une estimation locale du cout peut etre derivee a partir des `pricing_skus` publics quand la formule du modele est publiquement exploitable
- aucun test payant n'a encore ete execute

### Phase 3 - confort utilisateur

- menus dynamiques par modele
- support `frame_images` et `input_references`
- affichage du cout et des metadonnees
- meilleure gestion des erreurs

### Phase 4 - options avancees

- provider options specifiques
- support de l'analyse video en entree via `video_url` dans un node separe
- sauvegarde locale organisee des videos generees

## Recommandation immediate

La meilleure suite logique est:

1. corriger d'abord le **catalogue de modeles**
2. creer ensuite un **node video dedie**, sans toucher au flux stable du node chat
3. traiter **image-to-video** comme fonctionnalite de base du node video, pas comme option secondaire
4. ne faire les premiers tests payants qu'apres validation humaine explicite

Autrement dit:

- oui, on peut viser Wan, Seedance, Kling, Sora et Veo
- oui, on peut viser aussi le mode image-to-video et start/end-frame pour plusieurs modeles
- non, il ne faut pas essayer de les "faire rentrer" de force dans le node chat actuel sans refonte

## Sources

### Code du depot

- [node.py](K:\Codex_Projects\ComfyUI_OpenRouter\node.py)

### Documentation OpenRouter

- [Models](https://openrouter.ai/docs/guides/overview/models)
- [Chat Completions API](https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request)
- [Video Generation](https://openrouter.ai/docs/guides/overview/multimodal/video-generation)
- [Video Inputs](https://openrouter.ai/docs/guides/overview/multimodal/videos)
- [Provider Routing](https://openrouter.ai/docs/guides/routing/provider-selection)

### Documentation ComfyUI

- [GrokVideoNode](https://docs.comfy.org/built-in-nodes/GrokVideoNode)
- [Luma Text to Video](https://docs.comfy.org/built-in-nodes/partner-node/video/luma/luma-text-to-video)
- [Pika 2.2 Image to Video](https://docs.comfy.org/built-in-nodes/partner-node/video/pika/pika-image-to-video)

## Note de securite

Une cle API OpenRouter a ete fournie dans cette conversation, mais elle n'a pas ete utilisee dans le cadre de cette etude. Aucun test payant n'a ete lance.
