# Dataset and source review

Reviewed 10 November 2026. Provider terms and dataset records can change; verify rights for the intended use before collection.

| Source | Suitability | Implementation decision |
| --- | --- | --- |
| [Conceptual Captions on Hugging Face](https://huggingface.co/datasets/google-research-datasets/conceptual_captions) | Supplies `image_url` and `caption`; includes non-nature records, HTTP URLs, expired URLs, and third-party rights. Its dataset card labels the license “other.” | Read a bounded metadata page through the HF viewer API; nature-word filter; require provenance/rights note, HTTPS, approved host, validation, and manual review. No bulk download or blanket licensing claim. |
| [COCO](https://cocodataset.org/#termsofuse) / Flickr30k | Candidate paired-caption research sources, but local-image archives or dataset-specific access/rights arrangements are not a drop-in fit for strict URL-only storage. | No automatic archive download or invented URL reconstruction. Import independently verified URL metadata only. |
| [Pexels API](https://www.pexels.com/api/documentation/) | Official search metadata can provide source captions and photographer attribution. | API key and explicit ML permission required. Preserve original provider caption; search relevance alone never approves a pair. |
| [Pexels terms explanation](https://help.pexels.com/hc/en-us/articles/900005880463-What-are-the-Terms-and-Conditions) | Explicitly addresses permission for ML/AI dataset collection. | Collection disabled unless permission is documented and configured. |
| [Unsplash API](https://unsplash.com/documentation) and [terms](https://unsplash.com/terms) | Official API requires compliance with provider usage and attribution requirements. | Conservative explicit ML-permission gate; retain creator, source link, and original descriptive caption. No scraping. |

The app's batch limits are lower than typical provider pagination maxima. Requests respect HTTP failures and are never automatically expanded into large-scale collection. Provider permission may require additional contractual usage reporting; fulfill the actual agreement before using the adapter.

Pair audit: check scene identity, prominent nouns, colors, lighting, weather, spatial relationships, and licensing/provenance. Reject mismatches even if lexical overlap is high. Similarity signals are recorded separately; they are not calibrated semantic confidence. Group duplicates before splitting. Test/validation images must never contribute to tokenizer training.

The initial environment has about 15.7 GiB RAM and no `nvidia-smi` command. A modest CPU configuration can validate code and very small experiments; convincing from-scratch generation requires considerably more data and compute. Use the measured benchmark, not an unverified GPU-hour estimate, to budget experiments. Network fetch, optimizer, evaluation, and checkpoint time add to measured forward/backward time.
