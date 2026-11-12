# Evaluation rubric and limitations

Use a saved checkpoint, fixed prompt suite, seeds 42/123/2026, and fixed sampler/settings. Keep evaluation grids and their JSON settings with the experiment. Compare CFG values 1, 3, and 5 only after the unconditional branch has actually been trained.

Rate each generated sample independently:

| Score | Visual quality | Prompt adherence |
| --- | --- | --- |
| 1 | Unstructured noise or severe artifacts | No discernible relationship |
| 2 | Weak scene structure, major artifacts | Some broad colors or texture only |
| 3 | Recognizable scene, substantial defects | Main scene type fits, attributes missing |
| 4 | Coherent scene, minor defects | Most objects and attributes fit |
| 5 | Clear, convincing scene at native resolution | All assessable prompt attributes fit |

Document diversity across seeds, failure cases by category/lighting/weather, caption truncation/unknown words, and nearest training images using transient perceptual hashes plus manual comparison. Do not use pretrained feature extractors. Do not infer realism from low MSE. Record ambiguous ratings and reviewers' disagreements.

The baseline uses deterministic center-crop preprocessing to avoid creating crop/flip contradictions with captions. No image augmentation is applied. Pixel diffusion at 64×64 limits detail; human ratings must use the native resolution as well as any labeled display enlargement. CPU-only training is slow, URL sources can disappear, and a small dataset can cause memorization and weak conditioning.

## Initial local smoke run — 2026-11-10

At the time of training, five of ten Commons image URLs validated and were reviewed: two train pairs, two validation pairs, and one test pair. Five more returned HTTP 429 and were excluded from that run. A later bounded retry validated three additional URLs; eight of ten are now available, with the three newly available records still pending review and two URLs still returning HTTP 429. The checkpoint and metrics below refer to the original five approved pairs. Source images were fetched into RAM; only metadata and perceptual hashes were stored. The training records were selected with item-level CC0 or public-domain license labels and retain creator/source attribution.

Training used the CPU PyTorch 2.8.0 runtime and a deliberately small configuration: 64×64 resolution, two epochs, batch size 2, base channels 16, text dimension 64, one text layer, learning rate 0.0001, and seed 42. With only two training pairs, this produced one batch and one optimizer update per epoch: two updates total. The final training loss was 0.7211 and validation loss was 1.1279. These losses do not measure visual quality.

The saved checkpoint was restored for DDIM sampling. A 64×64 sample used the prompt “A dramatic waterfall in a lush tropical forest, mist rising from the rocks, bright morning sunlight, dense green leaves, realistic landscape photography.”, seed 42, guidance 4, and 50 sampling steps. The result at `backend/results/9860b501eb4644c9ac24e3acb6509535.png` is unstructured color noise: quality 1/5 and no discernible prompt adherence (1/5). Two other inspected samples from the same checkpoint, with different prompts, are similarly noisy. Increasing sampling steps cannot compensate for a model that has received only two parameter updates.

This confirms the training, checkpoint restore, and generation path runs end to end. It does **not** establish a usable generator, prompt conditioning, diversity, or resistance to memorization. The full fixed-prompt, multi-seed grid and broader real-data evaluation remain outstanding; this checkpoint should be treated as a mechanics demonstration only.
