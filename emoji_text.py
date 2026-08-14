"""CLIP text embeddings for emoji names — the conditioning bank, and prompt matching.

Two jobs, one frozen encoder:

  build_bank()  turns the dataset's emoji names into a (N+1, 512) tensor that
                TextConditionedUNet holds as a buffer and indexes like an
                nn.Embedding. The extra last row is the empty-prompt embedding,
                used as the null condition for classifier-free guidance.

  match()       embeds an arbitrary prompt and returns the nearest emoji names.
                Because it picks from a fixed list it cannot invent a name that
                does not exist, which is the failure mode a generative model
                would have here.

CLIP itself is never trained. The bank is computed once, cached to disk, and
frozen; only the projection inside the UNet learns anything.
"""

import torch
import torch.nn.functional as F

from metrics import ClipFeatures

NAME_PROMPT = "an emoji of {name}"

# openmoji ships keyword lists in its `description` column ("cheese, food,
# hungry, pepperoni, slice"). Folding them in pushes near-synonym names apart,
# which is the main weakness of a frozen text encoder on short labels.
KEYWORD_PROMPT = "an emoji of {name}. {description}"

NULL_PROMPT = ""


def name_prompts(names, descriptions=None):
    """Render the prompt string used to embed each emoji name."""
    if descriptions is None:
        return [NAME_PROMPT.format(name=n) for n in names]
    return [
        KEYWORD_PROMPT.format(name=n, description=d) if d and d.strip()
        else NAME_PROMPT.format(name=n)
        for n, d in zip(names, descriptions)
    ]


# Below this, the "embeddings" are near-orthogonal noise rather than language.
# Real CLIP text embeddings are strongly anisotropic — everything is somewhat
# similar to everything — so a mean pairwise cosine near 0 is diagnostic.
MIN_MEAN_COSINE = 0.15


def fingerprint(clip, names, descriptions=None):
    """Identifies *who* built a bank, not just what it covers.

    A cache keyed only on the name list will happily hand back vectors from a
    completely different encoder. That silently trained a whole model on random
    embeddings once; the fingerprint is why it cannot happen twice.
    """
    return {
        "model_name": getattr(clip, "model_name", "unknown"),
        "pretrained": getattr(clip, "pretrained", "unknown"),
        "prompt": KEYWORD_PROMPT if descriptions is not None else NAME_PROMPT,
        "n_names": len(names),
    }


def check_encoder(clip):
    """Fail loudly if the text encoder is not actually encoding meaning.

    Weights that failed to download, or a stub, still produce well-shaped
    tensors — everything downstream runs and returns numbers. This is the
    cheapest place to notice.
    """
    feats = clip.encode_texts(["a photo of a dog", "a photo of a puppy", "a photo of a rocket"])
    related = float(feats[0] @ feats[1])
    unrelated = float(feats[0] @ feats[2])

    if related < unrelated + 0.05:
        raise RuntimeError(
            "the CLIP text encoder is not producing meaningful embeddings: "
            f"cos(dog, puppy)={related:.3f} is not above cos(dog, rocket)={unrelated:.3f}. "
            "Pretrained weights probably failed to load — check open_clip and network access."
        )
    return {"related": related, "unrelated": unrelated}


def bank_health(bank):
    """Mean/max pairwise cosine over the bank — near 0 means random vectors."""
    rows = F.normalize(bank, dim=-1)
    sims = rows @ rows.T
    off = ~torch.eye(len(rows), dtype=torch.bool, device=rows.device)
    return {"mean_cosine": float(sims[off].mean()), "max_cosine": float(sims[off].max())}


def build_bank(names, descriptions=None, clip=None, device="cpu", cache_path=None):
    """Embed every name, plus a trailing null row. Returns (N + 1, dim).

    Pass `cache_path` to reuse a previous bank instead of reloading CLIP. The
    cache is keyed on the name list *and* a fingerprint of the encoder, so a
    bank built by a different (or broken) encoder is rebuilt rather than reused.
    """
    clip = clip or ClipFeatures(device=device)
    stamp = fingerprint(clip, names, descriptions)

    if cache_path is not None:
        cached = load_bank(cache_path, expect_names=names, expect_fingerprint=stamp)
        if cached is not None:
            return cached

    check_encoder(clip)

    prompts = name_prompts(names, descriptions) + [NULL_PROMPT]
    bank = clip.encode_texts(prompts).cpu()

    health = bank_health(bank)
    if health["mean_cosine"] < MIN_MEAN_COSINE:
        raise RuntimeError(
            f"the bank looks random: mean pairwise cosine {health['mean_cosine']:.3f} "
            f"< {MIN_MEAN_COSINE}. Real CLIP text embeddings sit far higher."
        )

    if cache_path is not None:
        torch.save({"names": list(names), "bank": bank, "fingerprint": stamp}, cache_path)
    return bank


def load_bank(cache_path, expect_names=None, expect_fingerprint=None):
    """Load a cached bank, or None if it is missing, stale, or foreign."""
    from pathlib import Path

    path = Path(cache_path)
    if not path.exists():
        return None

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if expect_names is not None and payload["names"] != list(expect_names):
        return None
    # A cache with no fingerprint predates this check — never trust it.
    if expect_fingerprint is not None and payload.get("fingerprint") != expect_fingerprint:
        return None
    return payload["bank"]


def encode_prompts(prompts, clip):
    """Embed free text into the same space as the bank (L2-normalized)."""
    if isinstance(prompts, str):
        prompts = [prompts]
    return clip.encode_texts(prompts).cpu()


def match(prompts, bank, names, clip, k=5):
    """Nearest emoji names for each prompt: [[(name, score), ...], ...].

    `bank` may include its trailing null row; it is ignored here.
    """
    bank = F.normalize(bank[: len(names)], dim=-1)
    query = F.normalize(encode_prompts(prompts, clip), dim=-1)

    scores, indices = (query @ bank.T).topk(min(k, len(names)), dim=-1)
    return [
        [(names[i], float(s)) for i, s in zip(row_i.tolist(), row_s.tolist())]
        for row_i, row_s in zip(indices, scores)
    ]


def retrieval_accuracy(bank, names, queries, targets, clip, ks=(1, 5, 10), batch_size=256):
    """Top-k accuracy of prompt -> name retrieval.

    With `queries` = the description keywords and `targets` = the matching
    names, this is a free labelled test set: every emoji carries its own query.
    """
    bank_n = F.normalize(bank[: len(names)], dim=-1)
    index = {name: i for i, name in enumerate(names)}
    gold = torch.tensor([index[t] for t in targets])

    ranks = []
    for start in range(0, len(queries), batch_size):
        chunk = queries[start : start + batch_size]
        query = F.normalize(encode_prompts(chunk, clip), dim=-1)
        order = (query @ bank_n.T).argsort(dim=-1, descending=True)
        want = gold[start : start + batch_size, None]
        ranks.append((order == want).float().argmax(dim=-1))
    ranks = torch.cat(ranks)

    out = {f"top{k}": float((ranks < k).float().mean()) for k in ks}
    out["mean_rank"] = float(ranks.float().mean() + 1)
    out["n_queries"] = len(queries)
    out["n_candidates"] = len(names)
    return out


def confusable_pairs(bank, names, top=15):
    """The most similar name pairs in the bank — the model's resolution limit.

    Two names at cosine ~0.99 are, to a frozen text encoder, the same
    instruction; no amount of training will let the UNet draw them apart.
    """
    bank_n = F.normalize(bank[: len(names)], dim=-1)
    sims = bank_n @ bank_n.T
    sims.fill_diagonal_(-1)

    best, partner = sims.max(dim=1)
    order = best.argsort(descending=True)

    seen, pairs = set(), []
    for i in order.tolist():
        j = int(partner[i])
        key = tuple(sorted((i, j)))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((names[i], names[j], float(best[i])))
        if len(pairs) >= top:
            break
    return pairs


if __name__ == "__main__":
    import csv
    from pathlib import Path

    from dataloader import OPENMOJI_500

    rows = list(csv.DictReader(open(Path(OPENMOJI_500) / "metadata.csv", encoding="utf-8")))
    names = [r["name"] for r in rows]
    descriptions = [r["description"] for r in rows]

    clip = ClipFeatures()
    print("encoder check:", check_encoder(clip))

    bank = build_bank(names, descriptions, clip=clip)
    print(f"bank: {tuple(bank.shape)} for {len(names)} names (+1 null row)")
    print("bank health:", bank_health(bank))

    # The keywords are the query, the name is the answer.
    labelled = [(r["description"], r["name"]) for r in rows if r["description"].strip()]
    queries, targets = zip(*labelled)
    print("retrieval:", retrieval_accuracy(bank, names, list(queries), list(targets), clip))

    for prompt in ("something to eat at a party", "a happy face", "fast transport"):
        hits = match(prompt, bank, names, clip, k=3)[0]
        print(f"  {prompt!r:<32} -> " + ", ".join(f"{n} ({s:.2f})" for n, s in hits))

    print("\nmost confusable pairs:")
    for a, b, s in confusable_pairs(bank, names, top=5):
        print(f"  {s:.3f}  {a}  ~  {b}")
