"""Optional, meeting-local speaker labels. Never owns capture or transcription.

Experimental thresholds require AMI development calibration before default enablement.
Stage 2 (cutting ASR input) is deliberately not enabled by this labelling pipeline.
"""
from collections import Counter, deque
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
import logging
import queue
import threading
import time

import numpy as np

from mellowd import models

log = logging.getLogger("mellowd.meeting_speakers")
RATE, WINDOW, STEP, CELL = 16000, 10., 2.5, .02
UNKNOWN = "Other participants"
MAX_FINGERPRINTS = 3000
# Conservative experimental starting points, NOT accuracy claims or synthetic tuning.
MATCH, MARGIN, UPDATE, UPDATE_MARGIN = .65, .12, .8, .2
CLUSTER_DISTANCE = .35
POWERSET = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                     [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.float32)


def session(path, *, optimize=True):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    if not optimize:
        # ORT 1.28.0 CPU basic/extended optimizers change this pinned CAM++ graph's
        # embeddings (same-voice cosine .38-.63 versus .90-.96 unoptimized).
        # Keep the reference graph until numerical parity is proven on an upgrade.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


class Models:
    def __init__(self, directory=None):
        import onnx_asr
        directory = Path(directory or models.MODELS_DIR)
        for name in models.SPEAKER_MODELS:
            if not models.verify(name, directory / name):
                raise RuntimeError("Speaker models are not verified. Prepare them before recording.")
        self.segmenter = session(directory / "meeting-segmentation.onnx")
        self.embedder = session(directory / "wespeaker_en_voxceleb_CAM++_LM.onnx", optimize=False)
        self.features = session(Path(onnx_asr.__file__).parent / "preprocessors/data/wespeaker.onnx")
        meta = self.segmenter.get_modelmeta().custom_metadata_map
        if (meta.get("sample_rate"), meta.get("window_size"), meta.get("num_classes")) != ("16000", "160000", "7"):
            raise RuntimeError("Unsupported segmentation model contract")

    def segment(self, audio):
        logits = self.segmenter.run(None, {"x": audio[None, None].astype(np.float32)})[0][0]
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        activity = probabilities @ POWERSET
        # Exact receptive-field centers from the pinned export, not duration / frames.
        centers = (np.arange(len(activity)) * 270 + 991 / 2) / RATE
        grid = np.arange(round(WINDOW / CELL)) * CELL + CELL / 2
        return np.stack([np.interp(grid, centers, activity[:, i]) for i in range(3)], axis=1)

    def fingerprint(self, audio):
        waveform = (np.clip(audio, -1, 32767 / 32768) * 32768).astype(np.float32)[None]
        feats, lengths = self.features.run(None, {"waveforms": waveform,
                                                  "waveforms_lens": np.array([waveform.shape[1]], dtype=np.int64)})
        feats = feats[:, :int(lengths[0])]
        # Follow WeSpeaker's infer_onnx.py: int16 scale, then temporal mean subtraction.
        # The plan's no-CMN probe was confounded by ORT graph optimization. With the
        # reference graph, omitting CMN makes volume changes look like new people.
        feats = feats - feats.mean(axis=1, keepdims=True)
        vector = self.embedder.run(None, {"feats": feats})[0][0]
        norm = np.linalg.norm(vector)
        if vector.shape != (512,) or not np.isfinite(vector).all() or norm < 1e-8:
            raise RuntimeError("Invalid speaker fingerprint")
        return (vector / norm).astype(np.float32)


def align(previous, current):
    """Return previous-index -> current-index mapping, scoring speech, not silence."""
    scores = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            a, b = previous[:, i], current[:, j]
            scores[i, j] = np.minimum(a, b).sum() / max(np.maximum(a, b).sum(), 1e-8)
    best = max(permutations(range(3)), key=lambda p: sum(scores[i, p[i]] for i in range(3)))
    return best, [scores[i, best[i]] if np.minimum(previous[:, i], current[:, best[i]]).sum() * CELL >= .15
                  else 0. for i in range(3)]


@dataclass
class Turn:
    start: float
    end: float
    track: int | None
    speaker: str = UNKNOWN
    overlap: bool = False
    vector: np.ndarray | None = None
    evidence: float = 0.


def cluster(vectors, threshold=CLUSTER_DISTANCE):
    """Average linkage via a nearest-neighbour chain: quadratic space and work.

    Independent branches can merge out of order; cutting the resulting tree rather
    than stopping at the first large distance is essential.
    """
    n = len(vectors)
    if not n:
        return []
    matrix = np.clip(1 - np.asarray(vectors) @ np.asarray(vectors).T, 0, 2)
    np.fill_diagonal(matrix, np.inf)
    active = np.ones(n, dtype=bool)
    sizes = np.ones(n)
    members = [[i] for i in range(n)]
    parent = list(range(n))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    chain = []
    while active.sum() > 1:
        if not chain:
            chain.append(int(np.flatnonzero(active)[0]))
        a = chain[-1]
        b = int(np.argmin(matrix[a]))
        # Prefer the previous link on ties, avoiding chains cycling among duplicates.
        if len(chain) > 1 and matrix[a, chain[-2]] <= matrix[a, b]:
            b = chain[-2]
        if len(chain) < 2 or b != chain[-2]:
            chain.append(b)
            continue
        if matrix[a, b] <= threshold:
            representative = root(members[a][0])
            for i in members[a] + members[b]:
                parent[root(i)] = representative
        weights = (matrix[a] * sizes[a] + matrix[b] * sizes[b]) / (sizes[a] + sizes[b])
        matrix[a] = weights
        matrix[:, a] = weights
        matrix[b] = matrix[:, b] = np.inf
        matrix[a, a] = np.inf
        sizes[a] += sizes[b]
        members[a].extend(members[b]); members[b] = []
        active[b] = False
        chain = chain[:-2]
    return [root(i) for i in range(n)]


def row_label(start, end, turns, *, timing_tolerance=0.):
    coverage = Counter()
    speech = 0.
    distinct = set()
    for turn in turns:
        amount = max(0., min(end, turn.end) - max(start, turn.start))
        if not amount:
            continue
        speech += amount
        if not turn.overlap and turn.speaker != UNKNOWN:
            coverage[turn.speaker] += amount
            distinct.add(turn.speaker)
    if not speech and timing_tolerance:
        # Decoder emission times are not phoneme boundaries. A short word just
        # outside activity can use the sole nearby identity, but never bridge two
        # different identities, unknown speech, or overlapping voices.
        neighbours = [t for t in turns if max(t.start - end, start - t.end, 0.) <= timing_tolerance]
        identities = {t.speaker for t in neighbours}
        if len(identities) == 1 and UNKNOWN not in identities and not any(t.overlap for t in neighbours):
            return identities.pop()
    if len(distinct) != 1 or not speech:
        return UNKNOWN
    key, amount = coverage.most_common(1)[0]
    return key if amount >= .8 * speech else UNKNOWN


class Timeline:
    """Worker-owned rolling audio, stitched windows and meeting-local vectors."""
    def __init__(self, engine):
        self.engine = engine
        self.audio = np.empty(0, dtype=np.float32)
        self.begin = self.end = self.next_window = None
        self.previous = None
        self.previous_tracks = []
        self.windows = deque()
        self.track_count = 0
        self.frontier = 0.
        self.early_frontier = 0.
        self.turns = []
        self.centroids = []
        self.current = None
        self.active_tracks = set()
        self.run_start = None
        self.pool_audio = []
        self.pool_turns = []
        self.pool_track = None

    def feed(self, audio, begin):
        if self.end is not None and abs(begin - self.end) > 2 / RATE:
            self.boundary()
        if self.begin is None:
            self.begin = self.end = begin
            # Fixed grid within each continuous capture epoch. Pauses reset it.
            self.next_window = begin
            self.frontier = begin
        self.audio = np.concatenate((self.audio, audio))
        self.end = begin + len(audio) / RATE
        while self.next_window + WINDOW <= self.end + 1 / RATE:
            self.window(self.next_window)
            self.next_window += STEP
        # No voice waveform older than 30 seconds survives.
        discard = max(0, len(self.audio) - 30 * RATE)
        if discard:
            self.audio = self.audio[discard:].copy()
            self.begin += discard / RATE

    def samples(self, begin, end):
        a = max(0, round((begin - self.begin) * RATE))
        b = max(a, round((end - self.begin) * RATE))
        return self.audio[a:b]

    def window(self, start):
        audio = self.samples(start, min(start + WINDOW, self.end))
        audio = np.pad(audio, (0, max(0, round(WINDOW * RATE) - len(audio))))
        activity = self.engine.segment(audio)
        tracks = [-1] * 3
        shift = round(STEP / CELL)
        if self.previous is not None:
            permutation, agreement = align(self.previous[shift:], activity[:-shift])
            for i, j in enumerate(permutation):
                if agreement[i] >= .45:
                    tracks[j] = self.previous_tracks[i]
        for i in range(3):
            if tracks[i] < 0:
                tracks[i] = self.track_count
                self.track_count += 1
        self.previous, self.previous_tracks = activity, tracks
        self.windows.append((start, activity, tracks))
        self.early_frontier = min(self.end, start + WINDOW - STEP)
        self.emit(min(start, self.end))
        while self.windows and self.windows[0][0] + WINDOW <= self.frontier:
            self.windows.popleft()

    def emit(self, until):
        while self.frontier + CELL <= until + 1e-6:
            center = self.frontier + CELL / 2
            scores = Counter()
            count = 0
            for start, activity, tracks in self.windows:
                index = int((center - start) / CELL)
                if start <= center < start + WINDOW and 0 <= index < len(activity):
                    count += 1
                    for i, track in enumerate(tracks):
                        scores[track] += activity[index, i]
            active = {track for track, score in scores.items()
                      if score / max(count, 1) >= (.4 if track in self.active_tracks else .6)}
            self.active_tracks = active
            state = tuple(sorted(active))
            if state != self.current or (self.run_start is not None and center - self.run_start >= 8):
                self.close_run(self.frontier)
                self.current, self.run_start = state, self.frontier
            self.frontier += CELL

    def close_run(self, end):
        if self.current and self.run_start is not None and end > self.run_start:
            overlap = len(self.current) > 1
            turn = Turn(self.run_start, end, self.current[0] if not overlap and end - self.run_start >= .1 else None, overlap=overlap)
            self.collect_evidence(turn)
            self.turns.append(turn)
            self.prune()
        self.run_start = end

    def collect_evidence(self, turn):
        """Build one clean voice sample across brief pauses in the same aligned track.

        This cannot bridge another speaker, overlap, a capture pause, or a long gap.
        Short phrases share measured evidence; they never inherit a nearby label
        merely because it is nearby.
        """
        contiguous = (self.pool_turns and self.pool_track == turn.track
                      and turn.start - self.pool_turns[-1].end <= 1.5
                      and turn.end - self.pool_turns[0].start <= 10)
        if not contiguous:
            self.pool_audio.clear(); self.pool_turns.clear()
        self.pool_track = turn.track
        if turn.overlap or turn.track is None:
            return
        self.pool_audio.append(self.samples(turn.start, turn.end).copy())
        self.pool_turns.append(turn)
        seconds = sum(len(a) for a in self.pool_audio) / RATE
        if seconds < 2:
            return
        # Keep inference bounded and prefer several seconds over phoneme-sized samples.
        waveform = np.concatenate(self.pool_audio)[-8 * RATE:]
        vector = self.engine.fingerprint(waveform)
        turn.vector, turn.evidence = vector, len(waveform) / RATE
        self.assign(turn)
        for member in self.pool_turns:
            member.vector, member.evidence = vector, turn.evidence
            member.speaker = turn.speaker

    def assign(self, turn):
        duration = turn.evidence or turn.end - turn.start
        if self.centroids:
            scores = np.array([float(turn.vector @ v) for v in self.centroids])
            best = int(scores.argmax())
            second = float(np.partition(scores, -2)[-2]) if len(scores) > 1 else -1.
            if scores[best] >= MATCH and scores[best] - second >= MARGIN:
                turn.speaker = f"Speaker {best + 1}"
                if duration >= 2 and scores[best] >= UPDATE and scores[best] - second >= UPDATE_MARGIN:
                    vector = .9 * self.centroids[best] + .1 * turn.vector
                    self.centroids[best] = vector / np.linalg.norm(vector)
                return
            # Borderline matches remain unknown instead of seeding a duplicate voice.
            if scores[best] > MATCH - MARGIN:
                return
        if duration >= 2:
            self.centroids.append(turn.vector.copy())
            turn.speaker = f"Speaker {len(self.centroids)}"

    def prune(self):
        retained = [t for t in self.turns if t.vector is not None]
        if len(retained) <= MAX_FINGERPRINTS:
            return
        buckets = Counter(int(t.start // 60) for t in retained)
        crowded = max(buckets.values())
        victim = min((t for t in retained if buckets[int(t.start // 60)] == crowded), key=lambda t: t.end - t.start)
        vector = victim.vector
        for turn in retained:
            if turn.vector is vector:
                turn.vector = None

    def boundary(self):
        if self.begin is not None:
            # Right padding supplies context for the tail, but never extends the meeting.
            while self.next_window < self.end:
                self.window(self.next_window)
                self.next_window += STEP
            self.emit(self.end)
            self.close_run(self.end)
        self.audio = np.empty(0, dtype=np.float32)
        self.begin = self.end = self.next_window = None
        self.previous = None
        self.previous_tracks = []
        self.windows.clear()
        self.current = self.run_start = None
        self.active_tracks.clear()
        self.pool_audio.clear(); self.pool_turns.clear(); self.pool_track = None

    def finalize(self):
        self.boundary()
        # Brief phrases can be repeatable but phonetic rather than speaker-specific.
        # They may match a final cluster, but cannot seed one or move its center.
        unique = {}
        for turn in self.turns:
            if turn.vector is not None and (turn.evidence or turn.end - turn.start) >= 2:
                unique.setdefault(id(turn.vector), turn)
        retained = list(unique.values())
        labels = cluster([t.vector for t in retained])
        for turn in self.turns:
            turn.speaker = UNKNOWN
        names = {}
        for turn, label in zip(retained, labels):
            names.setdefault(label, f"Speaker {len(names) + 1}")
            turn.speaker = names[label]
        shared = {id(t.vector): t.speaker for t in retained}
        for turn in self.turns:
            if turn.vector is not None and id(turn.vector) in shared:
                turn.speaker = shared[id(turn.vector)]
        centers = {}
        for label in names:
            vector = np.mean([t.vector for t, k in zip(retained, labels) if k == label], axis=0)
            centers[names[label]] = vector / max(np.linalg.norm(vector), 1e-8)
        for turn in self.turns:
            if turn.vector is not None and turn.speaker == UNKNOWN and centers:
                scores = sorted(((float(turn.vector @ v), key) for key, v in centers.items()), reverse=True)
                second = scores[1][0] if len(scores) > 1 else -1.
                if scores[0][0] >= MATCH and scores[0][0] - second >= MARGIN:
                    turn.speaker = scores[0][1]
        # Propagate only across contiguous pieces of the same stitched track.
        for sequence in (self.turns, list(reversed(self.turns))):
            for a, b in zip(sequence, sequence[1:]):
                adjacent = min(abs(a.end - b.start), abs(b.end - a.start)) <= CELL * 1.5
                if adjacent and a.track is not None and a.track == b.track and b.speaker == UNKNOWN:
                    b.speaker = a.speaker
        ordered_names = {}
        for turn in self.turns:
            if turn.speaker != UNKNOWN:
                ordered_names.setdefault(turn.speaker, f"Speaker {len(ordered_names) + 1}")
                turn.speaker = ordered_names[turn.speaker]
        result = self.snapshot()
        self.clear()
        return result

    def snapshot(self):
        return [Turn(t.start, t.end, t.track, t.speaker, t.overlap) for t in self.turns]

    def clear(self):
        self.audio = np.empty(0, dtype=np.float32)
        for turn in self.turns:
            turn.vector = None
        self.turns.clear(); self.centroids.clear(); self.windows.clear()
        self.pool_audio.clear(); self.pool_turns.clear(); self.pool_track = None
        self.previous = None
        self.engine = None


class Worker:
    """Capture callback only enqueues; any failure disables labels, never recording."""
    def __init__(self, engine=None, on_failure=None):
        self.timeline = Timeline(engine or Models())
        self.on_failure = on_failure or (lambda message: None)
        self.queue = queue.Queue(maxsize=300)  # 30s of Capture's 100ms frames
        self.failed = threading.Event()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.result = []
        self.frontier = 0.
        self.final = False
        self.cpu_seconds = 0.
        self.finalize_seconds = 0.
        self.thread = threading.Thread(target=self.run, name="meeting-speakers", daemon=True)
        self.thread.start()

    def feed(self, who, data, begin):
        if who != UNKNOWN or self.failed.is_set() or self.stop.is_set():
            return
        try:
            # Capture gives ownership of a new array; no copy or inference here.
            self.queue.put_nowait((data, begin))
        except queue.Full:
            self.failed.set()

    def boundary(self):
        if not self.failed.is_set():
            try:
                self.queue.put_nowait(None)
            except queue.Full:
                self.failed.set()

    def snapshot(self):
        with self.lock:
            return self.frontier, list(self.result)

    def run(self):
        last_publish = time.monotonic()
        cpu_start = time.thread_time()
        try:
            while not self.stop.is_set() or not self.queue.empty():
                if self.stop.is_set() and not self.final:
                    return
                if self.failed.is_set():
                    raise RuntimeError("Speaker processing fell behind")
                try:
                    item = self.queue.get(timeout=.1)
                except queue.Empty:
                    continue
                if item is None:
                    self.timeline.boundary()
                else:
                    self.timeline.feed(*item)
                if time.monotonic() - last_publish >= 1:
                    with self.lock:
                        self.frontier = self.timeline.frontier
                        self.result = self.timeline.snapshot()
                    last_publish = time.monotonic()
            if self.final and not self.failed.is_set():
                finalize_start = time.monotonic()
                result = self.timeline.finalize()
                self.finalize_seconds = time.monotonic() - finalize_start
                with self.lock:
                    self.result, self.frontier = result, float("inf")
        except Exception:
            self.failed.set()
            log.exception("Speaker labelling disabled; transcription continues")
            self.on_failure("Speaker labelling stopped. Transcription continues with available labels.")
        finally:
            self.cpu_seconds = time.thread_time() - cpu_start
            self.timeline.clear()
            while not self.queue.empty():
                self.queue.get_nowait()

    def close(self, final=False):
        self.final = final
        if not final:
            self.failed.set()
        self.stop.set()
        self.thread.join()
        result = self.snapshot()[1]
        log.info("speaker worker finished: final=%s failed=%s turns=%d worker_cpu=%.2fs finalize=%.2fs",
                 final, self.failed.is_set(), len(result), self.cpu_seconds, self.finalize_seconds)
        with self.lock:
            self.result = []
        return result
