# Independent Witnesses

A computational aesthetic movement built on a single forensic premise: any
measurement repeated across a system produces a chorus of local witnesses,
each carrying its own bias. The system's truth is never contained in any
single witness — it lives, latent and unobservable, in the consensus across
them. The algorithm's task is to make that dialectic visible: to put the raw
testimony of independent witnesses on the canvas alongside the consensus that
emerges when their voices are filtered through proximity.

The computational language is uncompromising and transparent. A known
ground-truth structure — three trees, one straight, one leaning, one
sinuously curved — is perturbed by realistic point-cloud noise. Each
horizontal slice of each tree is interrogated **independently**, with no
memory of its neighbours: a closed-form 2D circle is fit to the points in
that slice via the Kasa algebraic estimator, producing a witness point
`(X_c, Y_c)` at that height. The witnesses are then drawn twice — once as a
**raw polyline** that concatenates them faithfully, exposing every jitter and
every spasm of disagreement; and once as a **smoothed polyline** filtered
through a moving-median window that lets adjacent witnesses corroborate each
other. Beauty lives in the contrast between the two traces.

The aesthetic is forensic, not decorative. A dark inspection field on which
evidence emerges in measured colour: wood-toned scatter for the simulated
LiDAR returns, a faint white dashed line marking the unobservable true axis,
subtle blue rings hinting at each independent fit, magenta for the raw
testimony with all its contradictions, anthropic-green for the consensus
that smoothing reveals. There is no incidental ornament. Every glyph on the
screen earns its presence by carrying analytic weight.

The mathematical core is intentionally legible: Kasa's least-squares circle
fit is a 3×3 linear system, and the moving-median filter is a single pass
across the witness sequence. No black-box optimiser, no opaque library
call — the entire pipeline from noisy point to coherent centerline is
auditable on screen. This is the product of meticulous attention to
algorithmic clarity, the kind of implementation that emerges only when the
author cares enough to expose the machinery rather than hide it behind a
prettier surface.

Every parameter that affects the dialectic is exposed and reproducible:
noise level, witness density per slice, vertical resolution, smoothing
window, the lean angle of trunk two, the curve amplitude of trunk three.
Same seed, same scene, every time. The user perturbs the system and watches
the contradiction-versus-consensus dynamic respond in real time. The
craftsmanship lies precisely there: in making a piece of computational
geometry so transparent that the underlying epistemology — independent
witnesses cannot, on their own, tell you where the centre of a tree is —
becomes self-evident to anyone willing to look.
