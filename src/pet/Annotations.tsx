import { useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { useCoat } from "../ui/coatApply";
import { areaPathFor, arrowhead, arrowPathFor, dragPoint, isAreaMark, localPoint, pathFor, traceAt, traceLength, tracePoints, type AnnotationDisplay, type XY } from "./annotationScene";
import { strokeClock, strokeRemaining, strokeSequence, type Stroke } from "./annotationMotion";
import { captionObstacles, placeCaption } from "./annotationCaption";
import "./annotations.css";

/** Ink and bone share one animation clock; the pet owns the connection. */
export default function Annotations() {
  useCoat();
  const [display, setDisplay] = useState<AnnotationDisplay | null>(null);
  const [started, setStarted] = useState<number | null>(null);
  const svg = useRef<SVGSVGElement>(null), pen = useRef<SVGGElement>(null), caption = useRef<SVGGElement>(null);
  const finished = useRef(0);
  const currentRevision = useRef<number | null>(null);
  useEffect(() => {
    let disposed = false, changed = 0;
    const stop = listen<AnnotationDisplay | null>("annotation-scene", ({ payload }) => {
      if (!disposed) { changed++; currentRevision.current = payload?.revision ?? null; setDisplay(payload); }
    });
    const finish = listen<number>("annotation-pen-finished", ({ payload }) => {
      if (disposed || payload !== currentRevision.current) return;
      finished.current = payload;
      if (pen.current) pen.current.style.opacity = "0";
    });
    void stop.then(async () => {
      const seen = changed, current = await invoke<AnnotationDisplay | null>("annotation_snapshot");
      if (!disposed && seen === changed) { currentRevision.current = current?.revision ?? null; setDisplay(current); }
    }).catch((error) => console.error("[mellow] annotation subscription failed", error));
    return () => { disposed = true; for (const pending of [stop, finish]) void pending.then((off) => off()).catch(() => {}); };
  }, []);
  useEffect(() => {
    setStarted(null);
    if (!display) return;
    let cancelled = false, first = 0, second = 0;
    void invoke<boolean>("annotation_painted", { revision: display.revision, painted: false }).then((accepted) => {
      if (cancelled || !accepted) return;
      first = requestAnimationFrame(() => { second = requestAnimationFrame(() => {
        if (!cancelled) setStarted(display.revision);
      }); });
    }).catch((error) => console.error("[mellow] annotation placement failed", error));
    return () => { cancelled = true; cancelAnimationFrame(first); cancelAnimationFrame(second); };
  }, [display]);
  useEffect(() => {
    if (!display || started !== display.revision || !svg.current || !pen.current || !caption.current) return;
    const { scene, scale, revision } = display, carried = scene.reveal_from ?? 0;
    const reduced = matchMedia("(prefers-reduced-motion: reduce)");
    const context = document.createElement("canvas").getContext("2d");
    if (context) context.font = '14px "Segoe UI", sans-serif';
    const groups = Array.from(svg.current.querySelectorAll<SVGGElement>("[data-mark]"));
    const paths = groups.map((group) => group.querySelector<SVGPathElement>(".annotation-ink"));
    const traces = scene.marks.map((mark) => tracePoints(mark, display));
    const strokes: Stroke[] = [];
    for (let index = carried; index < scene.marks.length; index++) {
      const mark = scene.marks[index], path = paths[index];
      if (mark.kind === "label") continue;
      const area = isAreaMark(mark.kind), arrow = mark.kind === "arrow", drag = area || arrow;
      const trace = traces[index];
      const length = trace ? traceLength(trace) : path?.getTotalLength() ?? 0;
      const a = trace ? null : path?.getPointAtLength(0), b = trace ? null : path?.getPointAtLength(length);
      const start: XY = trace ? trace[0] : drag || !a ? localPoint(mark.points[0], display) : [a.x, a.y];
      const end: XY = trace ? trace[trace.length - 1] : drag || !b ? localPoint(mark.points[mark.points.length - 1], display) : [b.x, b.y];
      // Only a label before the next shape belongs to this shape.
      const following = scene.marks.slice(index + 1);
      const label = following.find((m, at) => m.kind === "label" && following.slice(0, at).every((p) => p.kind === "label"));
      strokes.push({ index, length: drag ? Math.hypot(end[0] - start[0], end[1] - start[1]) : length,
        start, end, gesture: area ? "area" : arrow ? "arrow" : undefined, caption: label?.text ?? scene.caption ?? "" });
    }
    if (!strokes.length) {
      const index = Math.min(carried, scene.marks.length - 1), mark = scene.marks[index], anchor = localPoint(mark.points[0], display);
      strokes.push({ index, start: anchor, end: anchor, length: 0, caption: mark.text ?? scene.caption ?? "" });
    }
    const steps = strokeSequence(strokes, display.pen_origin ? localPoint(display.pen_origin, display) : strokes[0].start);
    const bounds = scene.window ?? [scene.monitor.left, scene.monitor.top, scene.monitor.width, scene.monitor.height];
    const [left, top] = localPoint([bounds[0], bounds[1]], display);
    const [right, bottom] = localPoint([bounds[0] + bounds[2], bounds[1] + bounds[3]], display);
    const obstacles = captionObstacles(scene.marks.map((mark) => ({ ...mark,
      points: mark.points.map((point) => localPoint(point, display)) })));
    let cancelled = false, frame = 0, receiptFrame = 0, ready = false, readyConfirmed = false, completed = false;
    let start = -1, lastPose = -1000, activeCaption = "", boxWidth = 0;
    const readyPainted = (remainingMs: number) => {
      if (ready || cancelled) return;
      ready = true;
      receiptFrame = requestAnimationFrame(() => { receiptFrame = requestAnimationFrame(() => {
        if (!cancelled) void invoke<boolean>("annotation_painted", { revision, painted: true, remainingMs }).then((accepted) => {
          if (cancelled || !accepted) return;
          readyConfirmed = true;
          if (completed && !cancelled) void invoke("annotation_complete", { revision });
        }).catch((error) => console.error("[mellow] annotation receipt failed", error));
      }); });
    };
    const captionBox = (tip: XY) => placeCaption(tip, boxWidth, [left, top, right, bottom], obstacles);
    const showCaption = (text: string, tip: XY, visible: boolean) => {
      const node = caption.current!;
      if (text !== activeCaption) {
        activeCaption = text;
        let readable = text;
        const textWidth = Math.max(8, Math.min(194, right - left - 32));
        while (readable.length && (context?.measureText(readable).width ?? readable.length * 9) > textWidth) readable = readable.slice(0, -1);
        if (readable !== text) readable = readable.trimEnd() + "…";
        boxWidth = Math.min(Math.max(48, (context?.measureText(readable).width ?? readable.length * 9) + 20), Math.max(1, right - left - 12));
        node.querySelector("rect")!.setAttribute("width", String(boxWidth));
        node.querySelector("text")!.textContent = readable;
      }
      const box = captionBox(tip), readable = Boolean(visible && text && box);
      if (box) node.setAttribute("transform", `translate(${box[0] - tip[0]},${box[1] - tip[1]})`);
      node.style.opacity = readable ? "1" : "0";
      return readable;
    };
    const drawFrame = (timestamp: number) => {
      if (cancelled) return;
      if (start < 0) start = timestamp;
      const elapsed = reduced.matches ? Infinity : timestamp - start;
      const cursor = strokeClock(steps, elapsed), step = steps[cursor.index];
      const path = paths[step.index];
      const pointAt = (progress: number): XY => {
        if (step.gesture) return dragPoint(step.start, step.end, progress);
        const trace = traces[step.index];
        if (trace) return traceAt(trace, progress).tip;
        const point = path?.getPointAtLength(progress * step.length);
        return point ? [point.x, point.y] : step.end;
      };
      const tip: XY = cursor.phase === "draw" ? pointAt(cursor.progress) : cursor.tip;
      for (let index = 0; index < steps.length; index++) {
        const group = groups[steps[index].index];
        const progress = index < cursor.index || cursor.phase === "done" ? 1 : index === cursor.index && (cursor.phase === "draw" || cursor.phase === "hold") ? cursor.progress : 0;
        const area = steps[index].gesture === "area", arrow = steps[index].gesture === "arrow", drag = area || arrow;
        const trace = traces[steps[index].index];
        if (trace) {
          const d = traceAt(trace, progress).path;
          for (const ink of group.querySelectorAll<SVGPathElement>(".annotation-stroke")) ink.setAttribute("d", d);
        }
        if (area) {
          const d = areaPathFor(scene.marks[steps[index].index], display, progress);
          for (const shape of group.querySelectorAll<SVGPathElement>("path")) shape.setAttribute("d", d);
        }
        if (arrow) {
          const mark = scene.marks[steps[index].index];
          const d = arrowPathFor(mark, display, progress);
          for (const shaft of group.querySelectorAll<SVGPathElement>(".annotation-stroke")) shaft.setAttribute("d", d);
          for (const head of group.querySelectorAll<SVGPathElement>(".annotation-arrowhead")) {
            head.setAttribute("d", arrowhead(mark, display, progress));
            head.style.opacity = progress > 0 ? "1" : "0";
          }
        }
        for (const ink of group.querySelectorAll<SVGPathElement>(".annotation-stroke")) {
          ink.style.strokeDasharray = trace ? "none" : "1";
          ink.style.strokeDashoffset = drag || trace ? "0" : String(1 - progress);
          ink.style.opacity = progress > 0 ? "1" : "0";
        }
        for (const end of group.querySelectorAll<SVGElement>(".annotation-fill")) end.style.opacity = (area ? progress > 0 : progress >= 1) ? "1" : "0";
      }
      pen.current!.style.transform = `translate(${tip[0]}px,${tip[1]}px)`;
      pen.current!.style.opacity = finished.current === revision ? "0" : "1";
      // A quick selection drag settles before its caption appears, so the text
      // is readable instead of racing diagonally across the target.
      const captionVisible = cursor.phase === "done" || cursor.phase === "hold" || (cursor.phase === "draw" && !step.gesture);
      const captionPainted = showCaption(step.caption, tip, captionVisible);
      if (timestamp - lastPose > 100 || cursor.phase === "done") {
        lastPose = timestamp;
        // Cover the caption's bounded footprint around this sample; no large
        // path-area mask or screenshot is sent through the local IPC channel.
        const tips: XY[] = !captionPainted ? [] : [-150, 0, 150].map((delta) => {
          const p = Math.max(0, Math.min(1, cursor.progress + delta / Math.max(1, step.draw)));
          return pointAt(p);
        });
        const rects = tips.flatMap((at) => {
          const box = captionBox(at);
          if (!box) return [];
          const [x,y,w,h] = box;
          return [[(x - 4) * scale + scene.monitor.left, (y - 4) * scale + scene.monitor.top, (w + 8) * scale, (h + 8) * scale]];
        });
        void invoke("annotation_pen_position", { revision, tip: [tip[0] * scale + scene.monitor.left, tip[1] * scale + scene.monitor.top], rects }).catch(() => {});
      }
      if (cursor.phase !== "travel") readyPainted(strokeRemaining(steps, elapsed));
      if (cursor.phase === "done") {
        completed = true;
        if (readyConfirmed) void invoke("annotation_complete", { revision });
      } else frame = requestAnimationFrame(drawFrame);
    };
    const changed = () => { if (!cancelled && !completed) { cancelAnimationFrame(frame); frame = requestAnimationFrame(drawFrame); } };
    reduced.addEventListener("change", changed);
    frame = requestAnimationFrame(drawFrame);
    return () => { cancelled = true; cancelAnimationFrame(frame); cancelAnimationFrame(receiptFrame); reduced.removeEventListener("change", changed); };
  }, [started, display]);
  if (!display) return null;
  const { scene, scale } = display, width = scene.monitor.width / scale, height = scene.monitor.height / scale, source = scene.window;
  const left = Math.max(0, ((source?.[0] ?? scene.monitor.left) - scene.monitor.left) / scale);
  const top = Math.max(0, ((source?.[1] ?? scene.monitor.top) - scene.monitor.top) / scale);
  const right = Math.min(width, ((source ? source[0] + source[2] : scene.monitor.left + scene.monitor.width) - scene.monitor.left) / scale);
  const bottom = Math.min(height, ((source ? source[1] + source[3] : scene.monitor.top + scene.monitor.height) - scene.monitor.top) / scale);
  return <svg ref={svg} className="annotations" width="100%" height="100%" viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
    <defs><clipPath id="annotation-source"><rect x={left} y={top} width={Math.max(0, right - left)} height={Math.max(0, bottom - top)} /></clipPath></defs>
    <g clipPath="url(#annotation-source)">
      {scene.marks.map((mark, index) => {
        const color = { mint: "#65e6bf", amber: "#ffd479", violet: "#c6abff" }[mark.color];
        if (mark.kind === "label") return <g key={index} data-mark={index} />;
        const d = pathFor(mark, display);
        return <g key={index} data-mark={index} className={index < (scene.reveal_from ?? 0) ? "annotation-kept" : ""} fill="none" strokeLinecap="round" strokeLinejoin="round">
          {isAreaMark(mark.kind) && <path className="annotation-fill" d={d} fill={`${color}${mark.kind === "highlight" ? "26" : "14"}`} />}
          <path className="annotation-stroke" d={d} pathLength={1} stroke="#14251f" strokeWidth={7} />
          <path className="annotation-stroke annotation-ink" d={d} pathLength={1} stroke={color} strokeWidth={3} />
          {mark.kind === "arrow" && <>
            <path className="annotation-arrowhead" d={arrowhead(mark, display)} stroke="#14251f" strokeWidth={7} />
            <path className="annotation-arrowhead" d={arrowhead(mark, display)} stroke={color} strokeWidth={3} />
          </>}
        </g>;
      })}
    </g>
    <g ref={pen} className="annotation-pen">
      <foreignObject x={-4} y={-4} width={24} height={24}><div className="annotation-bone" /></foreignObject>
      <g ref={caption} className="annotation-caption"><rect width={48} height={28} rx={7} /><text x={10} y={19} /></g>
    </g>
  </svg>;
}
