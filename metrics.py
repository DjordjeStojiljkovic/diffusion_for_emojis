"""Evaluation metrics for the class-conditional emoji runs.

Four questions, and the metrics that answer each:

  Do the samples look like the data?   FID, KID, CMMD, precision
  Do they cover the data?              recall, intra-class diversity
  Do they obey the class label?        CLIP score, CLIP zero-shot accuracy
  Are they merely copied?              nearest-neighbour similarity, LPIPS
                                       nearest-neighbour distance, pixel RMSE,
                                       perceptual-copy rate

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
        # Kept so a cached artifact can record which encoder produced it.
        self.model_name = model_name
        self.pretrained = pretrained

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


class LpipsFeatures:
    """LPIPS-style perceptual distance, in a form that supports full pairwise use.

    LPIPS is ``sum_l mean_hw || w_l * (x_l - y_l) ||^2`` over channel-normalized
    VGG activations. Folding each layer's 1/sqrt(HW) and its channel weights
    into the flattened activation makes that *exactly* a squared Euclidean
    distance, so an N x M matrix costs one `cdist` per layer instead of N*M
    forward passes — which is what makes nearest-neighbour and copy detection
    affordable at all.

    The reference `lpips` package's calibrated channel weights are used when it
    is installed. Otherwise channels are unweighted: still a reasonable
    perceptual distance, but *not* comparable to published LPIPS numbers, so
    `backend` records which one actually ran and evaluate() reports it.
    """

    # ReLU outputs of the five VGG16 blocks — the taps LPIPS reads from.
    LAYERS = (3, 8, 15, 22, 29)

    def __init__(self, device="cpu"):
        from torchvision.models import VGG16_Weights, vgg16

        self.device = device
        features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.model = features[: max(self.LAYERS) + 1].eval().to(device)
        self.model.requires_grad_(False)
        self.weights, self.backend = self._channel_weights(device)

    def _channel_weights(self, device):
        try:
            import lpips
        except ImportError:
            return None, "vgg16-unweighted"
        try:
            net = lpips.LPIPS(net="vgg", verbose=False)
            # The linear layer applies w to the *squared* difference, so the
            # factor folded into the feature vector is sqrt(w).
            weights = [
                lin.model[-1].weight.detach().flatten().clamp(min=0).sqrt().to(device)
                for lin in net.lins
            ]
            return weights, "lpips-vgg"
        except Exception:
            return None, "vgg16-unweighted"

    @torch.no_grad()
    def _vectors(self, images, layer_index, batch_size=32):
        """Per-image vectors whose squared distance is this layer's LPIPS term.

        One layer at a time: at 64x64 the first VGG block alone is 262k
        dimensions per image, so holding all five for both image sets at once
        would cost more memory than the rest of the evaluation combined.
        """
        target = self.LAYERS[layer_index]
        out = []
        for i in range(0, len(images), batch_size):
            h = _resize_normalize(
                images[i : i + batch_size].to(self.device), 64, IMAGENET_MEAN, IMAGENET_STD
            )
            for index, layer in enumerate(self.model):
                h = layer(h)
                if index == target:
                    break

            f = F.normalize(h, dim=1)
            if self.weights is not None:
                f = f * self.weights[layer_index].view(1, -1, 1, 1)
            hw = f.shape[-1] * f.shape[-2]
            out.append((f / hw**0.5).flatten(1).half().cpu())
        return torch.cat(out)

    @torch.no_grad()
    def distance_matrix(self, a, b, chunk=64):
        """(len(a), len(b)) matrix of perceptual distances.

        Features are cached at half precision, so distances carry ~1e-3 of
        relative error — two orders of magnitude below the gap between a
        reproduction and a merely similar sample, which is all this feeds.
        """
        out = torch.zeros(len(a), len(b))
        for layer_index in range(len(self.LAYERS)):
            va, vb = self._vectors(a, layer_index), self._vectors(b, layer_index)
            for i in range(0, len(a), chunk):
                out[i : i + chunk] += torch.cdist(
                    va[i : i + chunk].float(), vb.float()
                ).pow(2)
            del va, vb
        return out


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
    # Subsets must be smaller than the full set, or every "random subset" is the
    # same set reordered and the reported std is a meaningless 0.
    subset_size = min(subset_size or max(2, min(1000, n // 2)), n)
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


def name_retrieval(image_feats, bank, labels, ks=(1, 5, 10), n_names=None):
    """Ask CLIP to pick each sample's emoji name out of the whole vocabulary.

    The text-conditional twin of `zero_shot_accuracy`: instead of 8 groups the
    candidate set is every name the model was trained on, so chance is 1/N
    rather than 1/8 and the number is far harsher.

    `emoji_text.build_bank` appends a trailing null-prompt row. Pass
    `n_names=len(names)` (or a pre-sliced bank) so that row is not offered as a
    retrieval candidate — it would be a name the model can never be right about.
    """
    names_bank = bank if n_names is None else bank[:n_names]
    if len(names_bank) <= int(labels.max()):
        raise ValueError(
            f"bank has {len(names_bank)} name rows but labels go up to {int(labels.max())}"
        )

    sims = image_feats @ names_bank.T
    true = sims.gather(1, labels[:, None])
    # Rank of the correct name, 0 = top. Counting strictly-better candidates
    # avoids a full argsort over the vocabulary.
    rank = (sims > true).sum(dim=1)

    out = {"name_clip_score": clip_score(image_feats, names_bank[labels])}
    out.update({f"name_top{k}": float((rank < k).float().mean()) for k in ks})
    out["name_chance"] = 1 / len(names_bank)
    return out


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


def nearest_neighbour_distance(distances, exclude_self=False):
    """Distance from each row to its closest column, and which column that was.

    The distance twin of `nearest_neighbour`, which works in cosine similarity.
    `exclude_self` is for a set compared against itself, where every row would
    otherwise match itself at distance 0.
    """
    d = distances.clone()
    if exclude_self:
        d.fill_diagonal_(float("inf"))
    best = d.min(dim=1)
    return best.values, best.indices


def perceptual_copies(fake_nn, real_nn, percentile=1.0):
    """Share of samples sitting closer to a training image than the training set
    sits to itself — i.e. reproductions rather than samples.

    The threshold is calibrated on the real-real nearest-neighbour distribution
    rather than fixed, because "too close" depends on how tightly the training
    set clusters: a set full of near-duplicate emoji has a far lower natural
    floor than a diverse one, and a fixed threshold would call the former
    memorized and the latter clean regardless of the model. Taking the
    `percentile`-th percentile of the real distances means the rate reads as
    "this many times more copy-like than the tightest 1% of real pairs".

    Returns (rate, threshold).
    """
    threshold = float(torch.quantile(real_nn.float(), percentile / 100))
    return float((fake_nn <= threshold).float().mean()), threshold


def pixel_distance_matrix(a, b, chunk=256):
    """Per-pixel RMSE between two image sets.

    The crudest copy check, and worth having next to LPIPS: it catches a
    near-exact reproduction that a perceptual metric might forgive, and it needs
    no pretrained network, so it still works when torchvision is unavailable.

    cdist's default matrix-multiply path is avoided deliberately. It computes
    ||a-b||^2 as ||a||^2 + ||b||^2 - 2ab, which cancels catastrophically when
    the two images are nearly identical: at 64x64 it reports an RMSE of up to
    0.07 between an image and *itself*. That is exactly the regime copy
    detection cares about, so the slower exact kernel is the right trade.
    """
    a_flat, b_flat = a.flatten(1), b.flatten(1)
    out = torch.zeros(len(a), len(b))
    for i in range(0, len(a), chunk):
        out[i : i + chunk] = torch.cdist(
            a_flat[i : i + chunk], b_flat, compute_mode="donot_use_mm_for_euclid_dist"
        ) / a_flat.shape[1] ** 0.5
    return out


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
    real_labels=None,
    fake_images=None,
    fake_labels=None,
    class_names=None,
    device="cpu",
    inception=None,
    clip=None,
    kid_subsets=100,
    lpips=None,
    copy_percentile=1.0,
):
    """Run every metric and return one flat dict, ready for json.dump.

    `real_images` / `fake_images` are (N, 3, H, W) tensors in [-1, 1]; labels
    are (N,) int64 class indices; `class_names[i]` names class i. Pass reused
    `inception` / `clip` / `lpips` extractors to avoid reloading weights per
    run, or `lpips=False` to skip the perceptual block entirely.

    Leave the labels and `class_names` as None for an unconditional run: the
    distribution, diversity and copy metrics all still apply, and the
    conditioning ones are simply omitted rather than faked with a dummy class.
    """
    real_images, fake_images = real_images.cpu(), fake_images.cpu()
    real_labels = None if real_labels is None else real_labels.cpu()
    fake_labels = None if fake_labels is None else fake_labels.cpu()
    if class_names is None:
        real_labels = fake_labels = None

    inception = inception or InceptionFeatures(device)
    clip = clip or ClipFeatures(device=device)

    f_real, f_fake = inception(real_images).cpu(), inception(fake_images).cpu()
    c_real, c_fake = clip.encode_images(real_images).cpu(), clip.encode_images(fake_images).cpu()

    precision, recall = precision_recall(f_real, f_fake)
    kid_mean, kid_std = kid(f_real, f_fake, subsets=kid_subsets)
    nn_sims, _ = nearest_neighbour(c_fake, c_real, normalized=True)

    # diversity() returns a bare float when there are no labels to group by.
    if fake_labels is None:
        div_fake, div_fake_per_class = diversity(c_fake, normalized=True), {}
        div_real = diversity(c_real, normalized=True)
    else:
        div_fake, div_fake_per_class = diversity(c_fake, fake_labels, normalized=True)
        div_real, _ = diversity(c_real, real_labels, normalized=True)

    # Everything above is label-free. The conditioning metrics below need a
    # class per image, which an unconditional run does not have: there is no
    # requested class for an independent classifier to agree or disagree with.
    conditioning = {}
    if class_names is not None:
        c_text = clip.encode_texts([CLASS_PROMPT.format(n) for n in class_names]).cpu()
        accuracy, per_class_acc, _ = zero_shot_accuracy(c_fake, c_text, fake_labels)
        # A real-vs-real baseline for the class accuracy: CLIP is not a perfect
        # judge of 64x64 emoji, so the generated score is only readable against it.
        accuracy_real, _, _ = zero_shot_accuracy(c_real, c_text, real_labels)
        conditioning = {
            "clip_score": clip_score(c_fake, c_text[fake_labels]),
            "clip_score_real": clip_score(c_real, c_text[real_labels]),
            "clip_accuracy": accuracy,
            "clip_accuracy_real": accuracy_real,
            "clip_accuracy_per_class": {class_names[i]: v for i, v in per_class_acc.items()},
        }

    # Pixel-space copy detection: no pretrained net, so it always runs.
    px_fake_nn, _ = nearest_neighbour_distance(
        pixel_distance_matrix(fake_images, real_images)
    )
    px_real_nn, _ = nearest_neighbour_distance(
        pixel_distance_matrix(real_images, real_images), exclude_self=True
    )
    px_copy_rate, px_threshold = perceptual_copies(px_fake_nn, px_real_nn, copy_percentile)

    perceptual = {}
    if lpips is not False:
        lpips = lpips or LpipsFeatures(device)
        # Real-vs-real gives the baseline the copy threshold is calibrated on.
        lp_fake_nn, _ = nearest_neighbour_distance(
            lpips.distance_matrix(fake_images, real_images)
        )
        lp_real_nn, _ = nearest_neighbour_distance(
            lpips.distance_matrix(real_images, real_images), exclude_self=True
        )
        copy_rate, threshold = perceptual_copies(lp_fake_nn, lp_real_nn, copy_percentile)
        perceptual = {
            "lpips_backend": lpips.backend,
            "lpips_nn_mean": float(lp_fake_nn.mean()),
            "lpips_nn_min": float(lp_fake_nn.min()),
            "lpips_nn_mean_real": float(lp_real_nn.mean()),
            "perceptual_copy_rate": copy_rate,
            "perceptual_copy_threshold": threshold,
        }

    return {
        "n_real": len(real_images),
        "n_fake": len(fake_images),
        "fid": fid(f_real, f_fake),
        "kid_mean": kid_mean,
        "kid_std": kid_std,
        "cmmd": cmmd(c_real, c_fake),
        "precision": precision,
        "recall": recall,
        **conditioning,
        "nn_similarity_mean": float(nn_sims.mean()),
        "nn_similarity_max": float(nn_sims.max()),
        "pixel_nn_rmse_mean": float(px_fake_nn.mean()),
        "pixel_nn_rmse_min": float(px_fake_nn.min()),
        "pixel_nn_rmse_mean_real": float(px_real_nn.mean()),
        "pixel_copy_rate": px_copy_rate,
        "pixel_copy_threshold": px_threshold,
        **perceptual,
        "diversity_fake": div_fake,
        "diversity_real": div_real,
        **({"diversity_fake_per_class": {
            class_names[i]: v for i, v in div_fake_per_class.items()
        }} if div_fake_per_class else {}),
    }


if __name__ == "__main__":
    # Sanity-check the maths on synthetic features: identical distributions
    # should score ~0 distance, and shifting one should move every metric.
    torch.manual_seed(0)
    a = torch.randn(256, 64)
    b = torch.randn(256, 64)
    shifted = torch.randn(256, 64) + 2.0

    print(f"fid  same: {fid(a, b):8.3f}   shifted: {fid(a, shifted):8.3f}")
    kid_same, kid_std = kid(a, b)
    print(f"kid  same: {kid_same:8.4f}   shifted: {kid(a, shifted)[0]:8.4f}   (std {kid_std:.4f})")
    assert kid_std > 0, "subsets must differ, or the std is meaningless"
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

    # --- copy detection, on images rather than features ---------------------
    real_imgs = torch.randn(32, 3, 16, 16).clamp(-1, 1)
    # Half the "samples" are verbatim copies of training images, half are new.
    fake_imgs = torch.cat([real_imgs[:16], torch.randn(16, 3, 16, 16).clamp(-1, 1)])

    d_fr = pixel_distance_matrix(fake_imgs, real_imgs)
    d_rr = pixel_distance_matrix(real_imgs, real_imgs)
    assert d_fr.shape == (32, 32)
    assert torch.allclose(d_rr.diagonal(), torch.zeros(32), atol=1e-6), \
        "an image's pixel distance to itself must be 0"

    fake_nn, idx = nearest_neighbour_distance(d_fr)
    assert torch.equal(idx[:16], torch.arange(16)), "a copy's neighbour is its original"
    assert fake_nn[:16].max() < 1e-5, "copies should sit at distance ~0"
    assert fake_nn[16:].min() > 0.1, "novel samples should not"

    real_nn, _ = nearest_neighbour_distance(d_rr, exclude_self=True)
    assert real_nn.min() > 0, "exclude_self must drop the zero self-distance"

    rate, threshold = perceptual_copies(fake_nn, real_nn)
    print(f"\npixel NN: copies {fake_nn[:16].mean():.4f} | "
          f"novel {fake_nn[16:].mean():.4f} | real-real {real_nn.mean():.4f}")
    print(f"copy rate: {rate:.2f} (expected 0.50), threshold {threshold:.4f}")
    assert abs(rate - 0.5) < 1e-6, f"exactly half were copies, got {rate}"

    print("\nok")
