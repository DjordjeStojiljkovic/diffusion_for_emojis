"""Evaluation metrics for the class-conditional emoji runs.

Four questions, and the metrics that answer each:

  Do the samples look like the data?   FID, KID, CMMD, precision
  Do they cover the data?              recall, intra-class diversity
  Do they obey the class label?        CLIP score, CLIP zero-shot accuracy
  Are they merely copied?              nearest-neighbour similarity

A caveat worth repeating in any writeup: FID is badly biased below a few
thousand samples, and these runs have 10 and 200 images. The FID numbers here
are only meaningful *relative to each other* under an identical protocol —
never against published figures. KID and CMMD are far better behaved at this
scale, which is why they are computed alongside.

The feature extractors import torchvision / open_clip lazily, so the tensor
maths below can be imported and tested without them installed.
"""

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Used for both the CLIP score and the zero-shot class accuracy.
CLASS_PROMPT = "an emoji of {}"


def _resize_normalize(images, size, mean, std):
    """[-1, 1] images -> resized, channel-normalized batch for a pretrained net."""
    x = (images.clamp(-1, 1) + 1) / 2
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)
    mean = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x.clamp(0, 1) - mean) / std


class InceptionFeatures:
    """2048-d InceptionV3 pool features — the standard FID/KID feature space.

    torchvision's weights are not the original TF-Inception ones used by the
    reference FID implementation, so absolute values differ slightly from
    published numbers. Consistent within this project, which is what matters.
    """

    def __init__(self, device="cpu"):
        from torchvision.models import Inception_V3_Weights, inception_v3

        self.device = device
        self.model = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1, init_weights=False
        )
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(device)

    @torch.no_grad()
    def __call__(self, images, batch_size=50):
        out = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size].to(self.device)
            out.append(self.model(_resize_normalize(batch, 299, IMAGENET_MEAN, IMAGENET_STD)))
        return torch.cat(out).float()


class ClipFeatures:
    """CLIP image/text embeddings, L2-normalized. Used for CMMD and the two
    conditioning metrics."""

    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu"):
        import open_clip

        self.device = device
        self.model = open_clip.create_model(model_name, pretrained=pretrained)
        self.model.eval().to(device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.size = self.model.visual.image_size
        self.size = self.size[0] if isinstance(self.size, (tuple, list)) else self.size

    @torch.no_grad()
    def encode_images(self, images, batch_size=64):
        out = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size].to(self.device)
            feats = self.model.encode_image(
                _resize_normalize(batch, self.size, CLIP_MEAN, CLIP_STD)
            )
            out.append(F.normalize(feats.float(), dim=-1))
        return torch.cat(out)

    @torch.no_grad()
    def encode_texts(self, texts):
        tokens = self.tokenizer(list(texts)).to(self.device)
        return F.normalize(self.model.encode_text(tokens).float(), dim=-1)


# --- distribution distances -------------------------------------------------


def _sym_sqrt(m):
    """PSD square root via eigendecomposition (avoids a scipy dependency)."""
    vals, vecs = torch.linalg.eigh(m)
    return (vecs * vals.clamp(min=0).sqrt()) @ vecs.mT


def fid(real, fake):
    """Frechet distance between two Gaussians fitted to the feature sets.

    ||mu_r - mu_f||^2 + tr(C_r + C_f - 2 (C_r C_f)^(1/2)).
    """
    real, fake = real.double(), fake.double()
    mu_r, mu_f = real.mean(0), fake.mean(0)
    cov_r, cov_f = torch.cov(real.T), torch.cov(fake.T)

    # tr((C_r C_f)^1/2) computed through a symmetric form so eigh applies.
    sqrt_r = _sym_sqrt(cov_r)
    cross = torch.linalg.eigvalsh(sqrt_r @ cov_f @ sqrt_r).clamp(min=0).sqrt().sum()

    diff = (mu_r - mu_f).pow(2).sum()
    return float(diff + cov_r.trace() + cov_f.trace() - 2 * cross)


def kid(real, fake, subsets=100, subset_size=None, generator=None):
    """Kernel Inception Distance: unbiased MMD^2 with a degree-3 polynomial kernel.

    Unlike FID this is unbiased, so it stays meaningful with a few hundred
    samples. Returns (mean, std) over random subsets.
    """
    n = min(len(real), len(fake))
    subset_size = min(subset_size or n, n)
    if subset_size < 2:
        raise ValueError("need at least 2 samples per subset")

    d = real.shape[1]
    real, fake = real.double(), fake.double()

    def poly(a, b):
        return (a @ b.T / d + 1).pow(3)

    scores = []
    for _ in range(subsets):
        r = real[torch.randperm(len(real), generator=generator)[:subset_size]]
        f = fake[torch.randperm(len(fake), generator=generator)[:subset_size]]
        m = subset_size

        k_rr = poly(r, r).fill_diagonal_(0).sum() / (m * (m - 1))
        k_ff = poly(f, f).fill_diagonal_(0).sum() / (m * (m - 1))
        k_rf = poly(r, f).mean()
        scores.append(k_rr + k_ff - 2 * k_rf)

    scores = torch.stack(scores)
    return float(scores.mean()), float(scores.std())


def cmmd(real, fake, sigma=10.0, scale=1000.0):
    """CLIP-MMD: Gaussian-kernel MMD^2 on CLIP embeddings, x1000.

    Distribution-free and unbiased-ish at small N — the metric proposed as a
    replacement for FID precisely because FID's Gaussian assumption and its
    thousands-of-samples requirement do not hold in settings like this one.
    """
    real, fake = real.double(), fake.double()
    gamma = 1 / (2 * sigma**2)

    def rbf(a, b):
        return torch.exp(-gamma * torch.cdist(a, b).pow(2))

    value = rbf(real, real).mean() + rbf(fake, fake).mean() - 2 * rbf(real, fake).mean()
    return float(scale * value)


def precision_recall(real, fake, k=3):
    """Improved precision/recall (Kynkaanniemi et al.).

    Precision = share of generated samples inside the real manifold (fidelity).
    Recall    = share of real samples inside the generated manifold (coverage).
    Separating these tells a blurry-but-varied model from a sharp-but-collapsed
    one, which a single FID number cannot.
    """

    def radii(feats):
        d = torch.cdist(feats, feats)
        # k-th neighbour, skipping the point itself.
        return d.kthvalue(min(k + 1, len(feats)), dim=1).values

    real, fake = real.double(), fake.double()
    d_rf = torch.cdist(real, fake)

    precision = (d_rf <= radii(real)[:, None]).any(dim=0).double().mean()
    recall = (d_rf <= radii(fake)[None, :]).any(dim=1).double().mean()
    return float(precision), float(recall)


# --- conditioning, diversity, memorization ----------------------------------


def clip_score(image_feats, text_feats):
    """Mean cosine similarity between each sample and its class prompt, x100.

    Features must be L2-normalized and row-aligned (one text row per image).
    """
    return float(100 * (image_feats * text_feats).sum(-1).mean())


def zero_shot_accuracy(image_feats, class_text_feats, labels):
    """Ask CLIP which class each generated image belongs to.

    The most direct measure of whether conditioning worked: the model was told
    to draw class y, so an independent classifier should agree. Returns
    (accuracy, per-class accuracy, predictions).
    """
    preds = (image_feats @ class_text_feats.T).argmax(dim=1)
    correct = (preds == labels).float()

    per_class = {}
    for c in labels.unique().tolist():
        per_class[int(c)] = float(correct[labels == c].mean())
    return float(correct.mean()), per_class, preds


def nearest_neighbour(fake, real, normalized=False):
    """Cosine similarity of each generated sample to its closest training image.

    With 10 training images a diffusion model can simply reproduce them; a mean
    near 1.0 means the "generation" is recall, not synthesis. Returns
    (similarities, indices into `real`).
    """
    a = fake if normalized else F.normalize(fake, dim=-1)
    b = real if normalized else F.normalize(real, dim=-1)
    sims = a @ b.T
    best = sims.max(dim=1)
    return best.values, best.indices


def diversity(feats, labels=None, normalized=False):
    """Mean pairwise cosine distance, within class when labels are given.

    Near 0 means every sample of a class is the same picture — mode collapse,
    the characteristic failure of tiny-data conditional models.
    """
    x = feats if normalized else F.normalize(feats, dim=-1)

    def spread(f):
        if len(f) < 2:
            return float("nan")
        sims = f @ f.T
        off = ~torch.eye(len(f), dtype=torch.bool, device=f.device)
        return float(1 - sims[off].mean())

    if labels is None:
        return spread(x)

    per_class = {int(c): spread(x[labels == c]) for c in labels.unique()}
    values = [v for v in per_class.values() if v == v]  # drop NaNs
    return (sum(values) / len(values) if values else float("nan")), per_class


# --- the whole battery ------------------------------------------------------


def evaluate(
    real_images,
    real_labels,
    fake_images,
    fake_labels,
    class_names,
    device="cpu",
    inception=None,
    clip=None,
    kid_subsets=100,
):
    """Run every metric and return one flat dict, ready for json.dump.

    `real_images` / `fake_images` are (N, 3, H, W) tensors in [-1, 1]; labels
    are (N,) int64 class indices; `class_names[i]` names class i. Pass reused
    `inception` / `clip` extractors to avoid reloading weights per run.
    """
    real_images, fake_images = real_images.cpu(), fake_images.cpu()
    real_labels, fake_labels = real_labels.cpu(), fake_labels.cpu()

    inception = inception or InceptionFeatures(device)
    clip = clip or ClipFeatures(device=device)

    f_real, f_fake = inception(real_images).cpu(), inception(fake_images).cpu()
    c_real, c_fake = clip.encode_images(real_images).cpu(), clip.encode_images(fake_images).cpu()
    c_text = clip.encode_texts([CLASS_PROMPT.format(n) for n in class_names]).cpu()

    precision, recall = precision_recall(f_real, f_fake)
    kid_mean, kid_std = kid(f_real, f_fake, subsets=kid_subsets)
    accuracy, per_class_acc, _ = zero_shot_accuracy(c_fake, c_text, fake_labels)
    nn_sims, _ = nearest_neighbour(c_fake, c_real, normalized=True)
    div_fake, div_fake_per_class = diversity(c_fake, fake_labels, normalized=True)
    div_real, _ = diversity(c_real, real_labels, normalized=True)

    # A real-vs-real baseline for the class accuracy: CLIP is not a perfect
    # judge of 64x64 emoji, so the generated score is only readable against it.
    accuracy_real, _, _ = zero_shot_accuracy(c_real, c_text, real_labels)

    return {
        "n_real": len(real_images),
        "n_fake": len(fake_images),
        "fid": fid(f_real, f_fake),
        "kid_mean": kid_mean,
        "kid_std": kid_std,
        "cmmd": cmmd(c_real, c_fake),
        "precision": precision,
        "recall": recall,
        "clip_score": clip_score(c_fake, c_text[fake_labels]),
        "clip_score_real": clip_score(c_real, c_text[real_labels]),
        "clip_accuracy": accuracy,
        "clip_accuracy_real": accuracy_real,
        "clip_accuracy_per_class": {class_names[i]: v for i, v in per_class_acc.items()},
        "nn_similarity_mean": float(nn_sims.mean()),
        "nn_similarity_max": float(nn_sims.max()),
        "diversity_fake": div_fake,
        "diversity_real": div_real,
        "diversity_fake_per_class": {
            class_names[i]: v for i, v in div_fake_per_class.items()
        },
    }


if __name__ == "__main__":
    # Sanity-check the maths on synthetic features: identical distributions
    # should score ~0 distance, and shifting one should move every metric.
    torch.manual_seed(0)
    a = torch.randn(256, 64)
    b = torch.randn(256, 64)
    shifted = torch.randn(256, 64) + 2.0

    print(f"fid  same: {fid(a, b):8.3f}   shifted: {fid(a, shifted):8.3f}")
    print(f"kid  same: {kid(a, b)[0]:8.4f}   shifted: {kid(a, shifted)[0]:8.4f}")
    print(f"cmmd same: {cmmd(a, b):8.4f}   shifted: {cmmd(a, shifted):8.4f}")
    print("p/r  same: {:.2f}/{:.2f}   shifted: {:.2f}/{:.2f}".format(
        *precision_recall(a, b), *precision_recall(a, shifted)
    ))

    # A known closed form: for N(0,I) vs N(mu,I), FID -> ||mu||^2.
    big_a, big_mu = torch.randn(20000, 8), torch.randn(20000, 8) + 1.0
    print(f"\nfid vs closed form: {fid(big_a, big_mu):.3f} (expected ~{8 * 1.0**2:.1f})")

    labels = torch.arange(4).repeat_interleave(8)
    same = torch.randn(1, 16).repeat(32, 1)
    print(f"\ndiversity collapsed: {diversity(same, labels)[0]:.4f} (expected ~0)")
    print(f"diversity varied:    {diversity(torch.randn(32, 16), labels)[0]:.4f}")

    sims, idx = nearest_neighbour(a[:10], a)
    assert torch.equal(idx, torch.arange(10)), "a sample's nearest neighbour is itself"
    print(f"nn self-similarity:  {sims.mean():.4f} (expected 1.0)")

    text = F.normalize(torch.randn(4, 16), dim=-1)
    perfect = text[labels]
    acc, _, _ = zero_shot_accuracy(perfect, text, labels)
    print(f"zero-shot on perfect features: {acc:.2f} (expected 1.0)")
