/**
 * Checks the updater manifest before it reaches a release. A wrong one is
 * silent: the app just says it cannot check for updates.
 *
 *   node scripts/updater.check.ts                          # check the rules
 *   node scripts/updater.check.ts <latest.json> <version> <asset>
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

/** The only target Mellow ships. */
const PLATFORM = "windows-x86_64";

/** The plugin parses pub_date strictly as RFC3339 and rejects the whole manifest. */
const RFC3339 = /^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(\.\d+)?([Zz]|[+-]\d{2}:\d{2})$/;

const BASE64 = /^[A-Za-z0-9+/=\s]+$/;

export function problems(manifest: unknown, version: string, asset: string): string[] {
  const found: string[] = [];
  if (typeof manifest !== "object" || manifest === null) return ["manifest is not an object"];
  const m = manifest as Record<string, unknown>;

  if (m.version !== version) found.push(`version is ${JSON.stringify(m.version)}, expected "${version}"`);

  if (typeof m.pub_date !== "string" || !RFC3339.test(m.pub_date)) {
    found.push(`pub_date ${JSON.stringify(m.pub_date)} is not RFC3339`);
  }

  const platforms = m.platforms as Record<string, unknown> | undefined;
  if (typeof platforms !== "object" || platforms === null) {
    found.push("platforms is missing");
    return found;
  }
  const keys = Object.keys(platforms);
  if (keys.length !== 1 || keys[0] !== PLATFORM) {
    found.push(`platforms must be exactly { ${PLATFORM} }, got { ${keys.join(", ")} }`);
    return found;
  }

  const target = platforms[PLATFORM] as Record<string, unknown>;
  if (typeof target !== "object" || target === null) return [...found, `${PLATFORM} is not an object`];

  const url = target.url;
  if (typeof url !== "string" || !url.startsWith("https://")) {
    found.push(`url ${JSON.stringify(url)} is not an https URL`);
  } else if (!url.endsWith(`/${asset}`)) {
    found.push(`url does not point at the built asset "${asset}": ${url}`);
  } else if (!url.includes(`/download/v${version}/`)) {
    found.push(`url is not under the v${version} release: ${url}`);
  }

  const signature = target.signature;
  if (typeof signature !== "string" || signature.trim() === "") {
    found.push("signature is empty - was the build signed?");
  } else if (!BASE64.test(signature)) {
    found.push("signature is not base64; it must be the contents of the .sig file");
  }

  return found;
}

const [file, version, asset] = process.argv.slice(2);

if (file) {
  if (!version || !asset) {
    console.error("usage: node scripts/updater.check.ts <latest.json> <version> <asset>");
    process.exit(2);
  }
  const found = problems(JSON.parse(readFileSync(file, "utf8")), version, asset);
  if (found.length) {
    console.error(`${file} is not publishable:`);
    for (const problem of found) console.error(`  - ${problem}`);
    process.exit(1);
  }
  console.log(`updater manifest ok: ${version} -> ${asset}`);
} else {
  // No file given: check that each rule actually rejects.
  const VERSION = "1.2.3";
  const ASSET = "Mellow-Setup-1.2.3-x64.exe";
  const good = {
    version: VERSION,
    notes: "See the GitHub release.",
    pub_date: "2026-09-09T05:42:30.2408830Z",
    platforms: {
      [PLATFORM]: {
        signature: "dW50cnVzdGVkIGNvbW1lbnQ6IHNpZ25hdHVyZQo=",
        url: `https://github.com/Tarun-032/Mellow/releases/download/v${VERSION}/${ASSET}`,
      },
    },
  };

  assert.deepEqual(problems(good, VERSION, ASSET), [], "a correct manifest was rejected");

  /** One thing broken, and the word the complaint must mention. */
  const broken: [string, unknown, string][] = [
    ["wrong version", { ...good, version: "1.2.4" }, "version"],
    ["missing pub_date", { ...good, pub_date: undefined }, "pub_date"],
    ["loose date", { ...good, pub_date: "2026-09-09 05:42:30" }, "pub_date"],
    ["no platforms", { ...good, platforms: undefined }, "platforms"],
    ["wrong platform", { ...good, platforms: { "linux-x86_64": good.platforms[PLATFORM] } }, "platforms"],
    ["extra platform", { ...good, platforms: { ...good.platforms, "darwin-aarch64": {} } }, "platforms"],
    ["http url", { ...good, platforms: { [PLATFORM]: { ...good.platforms[PLATFORM], url: `http://x/v${VERSION}/${ASSET}` } } }, "https"],
    ["stale asset name", { ...good, platforms: { [PLATFORM]: { ...good.platforms[PLATFORM], url: `https://github.com/Tarun-032/Mellow/releases/download/v${VERSION}/Mellow-Setup-1.0.0-x64.exe` } } }, "asset"],
    ["url from another release", { ...good, platforms: { [PLATFORM]: { ...good.platforms[PLATFORM], url: `https://github.com/Tarun-032/Mellow/releases/download/v9.9.9/${ASSET}` } } }, "release"],
    ["unsigned", { ...good, platforms: { [PLATFORM]: { ...good.platforms[PLATFORM], signature: "" } } }, "signature"],
    ["signature is a path", { ...good, platforms: { [PLATFORM]: { ...good.platforms[PLATFORM], signature: "C:\keys\mellow.sig" } } }, "base64"],
    ["not an object", "latest.json", "object"],
  ];

  for (const [name, manifest, mentions] of broken) {
    const found = problems(manifest, VERSION, ASSET);
    assert.ok(found.length > 0, `${name} was accepted`);
    assert.ok(
      found.some((p) => p.toLowerCase().includes(mentions)),
      `${name} complained about the wrong thing: ${found.join("; ")}`,
    );
  }

  console.log("updater manifest rules: ok");
}
