export type Spring = {
  value: number;
  velocity: number;
  target: number;
};

export const spring = (value: number): Spring => ({
  value,
  velocity: 0,
  target: value,
});

/** Stable implicit spring step. `frequency` is the only feel knob. */
export function stepSpring(
  s: Spring,
  dt: number,
  frequency: number,
  damping = 1,
) {
  const omega = Math.PI * 2 * frequency;
  const f = 1 + 2 * dt * damping * omega;
  const oo = omega * omega;
  const hoo = dt * oo;
  const hhoo = dt * hoo;
  const inv = 1 / (f + hhoo);
  const value = s.value;
  const velocity = s.velocity;
  s.value = (f * value + dt * velocity + hhoo * s.target) * inv;
  s.velocity = (velocity + hoo * (s.target - value)) * inv;
}
