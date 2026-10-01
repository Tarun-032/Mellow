export type GuideOutcome = "arrived" | "failed";
export type GuideAck = { accepted: boolean; arrived: boolean };

/** Correlate the native revision with one backend presentation. */
export class GuidePresentation {
  private active: {
    revision: number;
    report: (outcome: GuideOutcome) => void;
    accepted: boolean;
    arrived: boolean;
    settled: boolean;
  } | null = null;

  start(revision: number, report: (outcome: GuideOutcome) => void) {
    this.active = { revision, report, accepted: false, arrived: false, settled: false };
  }

  clear() { this.active = null; }

  arrived(revision: number) {
    const active = this.active;
    if (!active || active.revision !== revision) return;
    active.arrived = true;
    if (active.accepted) this.finish(revision, "arrived");
  }

  result(revision: number, ack: GuideAck) {
    const active = this.active;
    if (!active || active.revision !== revision) return;
    if (!ack.accepted) { this.finish(revision, "failed"); return; }
    active.accepted = true;
    if (ack.arrived || active.arrived) this.finish(revision, "arrived");
  }

  failed(revision: number) { this.finish(revision, "failed"); }

  private finish(revision: number, outcome: GuideOutcome) {
    const active = this.active;
    if (!active || active.revision !== revision || active.settled) return;
    active.settled = true;
    active.report(outcome);
  }
}
