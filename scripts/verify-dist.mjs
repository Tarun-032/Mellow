import { readFile, readdir, stat } from "node:fs/promises";
import { extname, join, relative, resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const dist = join(root, "dist");
const textExtensions = new Set([".css", ".html", ".js"]);
const assetPattern = /(?:src|href)=["']\/(?!\/)|url\(\s*["']?\/(?!\/)|(?:import|from)\s*\(?\s*["']\/(?!\/)|["'`]\/assets\//;
const referencedAssets = new Set();

async function filesUnder(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) files.push(...(await filesUnder(path)));
    else files.push(path);
  }
  return files;
}

const files = await filesUnder(dist);
const violations = [];
for (const file of files) {
  if (!textExtensions.has(extname(file))) continue;
  const source = await readFile(file, "utf8");
  if (assetPattern.test(source)) {
    violations.push(relative(root, file));
  }

  // Bundled url() targets must exist; skip data/http/#.
  for (const match of source.matchAll(/url\(\s*["']?([^"')]+)["']?\s*\)/g)) {
    const value = match[1];
    if (/^(?:data:|https?:|#)/i.test(value)) continue;
    const target = resolve(file, "..", value.split(/[?#]/, 1)[0]);
    referencedAssets.add(target);
  }
}

if (violations.length) {
  throw new Error(
    `Packaged frontend contains root-relative asset URLs: ${violations.join(", ")}`,
  );
}

const missing = [];
for (const asset of referencedAssets) {
  try {
    if (!(await stat(asset)).isFile()) missing.push(relative(root, asset));
  } catch {
    missing.push(relative(root, asset));
  }
}
if (missing.length) {
  throw new Error(`Packaged frontend references missing assets: ${missing.join(", ")}`);
}

const requiredKinds = [".css", ".js", ".png", ".woff2"];
for (const extension of requiredKinds) {
  if (!files.some((file) => extname(file) === extension)) {
    throw new Error(`Packaged frontend has no ${extension} asset`);
  }
}

console.log(`Verified ${files.length} packaged frontend files with relocatable asset URLs.`);
