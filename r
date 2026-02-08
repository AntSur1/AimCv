#!/usr/bin/env node
import fs from "fs";
import path from "path";

const POSTS_DIR = "./posts";
const MAIN_JS = "./main.js";

/* ---------- helpers ---------- */

function slugify(str) {
  return str
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");
}

function getToday() {
  const d = new Date();
  return d.toISOString().split("T")[0]; // YYYY-MM-DD
}

function parseFrontmatter(text, file) {
  const match = text.match(/---([\s\S]*?)---/);
  if (!match) throw new Error(`Missing front-matter in ${file}`);

  const meta = {};
  for (const line of match[1].split("\n")) {
    const [key, ...rest] = line.split(":");
    if (!key || !rest.length) continue;
    meta[key.trim()] = rest.join(":").trim();
  }

  if (!meta.date || !meta.title)
    throw new Error(`Missing date or title in ${file}`);

  return meta;
}

/* ---------- main ---------- */

const mode = process.argv[2] || "sync";

if (mode === "new" || mode === "n") {
  // Create a template file with today's date
  let tempName = "temp.md";
  let counter = 1;

  while (fs.existsSync(path.join(POSTS_DIR, tempName))) {
    tempName = `temp-${counter}.md`;
    counter++;
  }

  const today = getToday();
  const template = `---
title: Title
date: ${today}
---

Write your devlog here.
`;

  fs.writeFileSync(path.join(POSTS_DIR, tempName), template, "utf8");
  console.log(`✔ Created new file: ${tempName} (date: ${today})`);
} 
else if (mode === "sync" || mode === "s") {
  // Step 1: read all md files
  const mdFiles = fs
    .readdirSync(POSTS_DIR)
    .filter(f => f.endsWith(".md"));

  let finalFiles = [];

  // Step 2: rename files based on frontmatter
  for (const file of mdFiles) {
    const fullPath = path.join(POSTS_DIR, file);
    const text = fs.readFileSync(fullPath, "utf8");

    let date, title;
    try {
      ({ date, title } = parseFrontmatter(text, file));
    } catch (e) {
      console.warn(`Skipping ${file}: ${e.message}`);
      continue;
    }

    const slug = slugify(title);
    const newName = `${date}-${slug}.txt`;
    const newPath = path.join(POSTS_DIR, newName);

    if (file !== newName) {
      if (fs.existsSync(newPath)) {
        console.warn(`⚠ Target file already exists, skipping rename: ${newName}`);
        finalFiles.push(newName);
        continue;
      }
      fs.renameSync(fullPath, newPath);
      console.log(`renamed: ${file} → ${newName}`);
    }

    finalFiles.push(newName);
  }

  // Step 3: update main.js
  let mainJs = fs.readFileSync(MAIN_JS, "utf8");
  const match = mainJs.match(/const postFiles\s*=\s*\[([\s\S]*?)\];/);

  if (!match) throw new Error("Could not find postFiles array in main.js");

  const existing = match[1]
    .split("\n")
    .map(l => l.trim())
    .filter(Boolean)
    .map(l => l.replace(/["',]/g, ""));

  const merged = Array.from(new Set([...existing, ...finalFiles])).sort();

  const newArray =
    "const postFiles = [\n" +
    merged.map(f => `  "${f}",`).join("\n") +
    "\n];";

  mainJs = mainJs.replace(/const postFiles\s*=\s*\[[\s\S]*?\];/, newArray);
  fs.writeFileSync(MAIN_JS, mainJs);
  console.log(`✔ Synced ${merged.length} posts`);
} 
else {
  console.log("Unknown mode. Use: node r (n)ew | node r (s)ync");
}
