const POSTS_DIR = "posts";
const entriesEl = document.getElementById("entries");

/**
 * You must list files manually or auto-generate this list via a build step.
 */
const postFiles = [
  "2026-02-08-previously.txt",
];

async function loadPost(filename) {
  try {
    const res = await fetch(`./${POSTS_DIR}/${filename}`);
    
    if (!res.ok) {
      console.warn(`Skipping ${filename}: ${res.status}`);
      return null; // skip missing files
    }

    const text = await res.text();

    // safely parse frontmatter
    const cleanText = text.replace(/^\uFEFF/, ""); // strip possible BOM
    const lines = cleanText.split(/\r?\n/);

    if (lines[0].trim() !== "---") {
      console.warn(`Skipping ${filename}: invalid frontmatter`);
      return null;
    }

    let i = 1;
    const frontmatterLines = [];

    while (i < lines.length && lines[i].trim() !== "---") {
      frontmatterLines.push(lines[i]);
      i++;
    }

    if (i >= lines.length) {
      console.warn(`Skipping ${filename}: invalid frontmatter`);
      return null;
    }

    const frontmatter = frontmatterLines.join("\n");
    const content = lines.slice(i + 1).join("\n");

    const meta = Object.fromEntries(
      frontmatter
        .trim()
        .split("\n")
        .map(line => {
          const parts = line.split(":");
          if (parts.length < 2) return ["", ""];
          return parts.map(s => s.trim());
        })
        .filter(([key]) => key) // remove empty lines
    );

    return {
      title: meta.title || filename,
      date: meta.date || "",
      html: marked.parse(content)
    };

  } catch (err) {
    console.error(`Error loading ${filename}:`, err);
    return null; // skip on network or parsing error
  }
}

async function loadAll() {
  const posts = await Promise.all(postFiles.reverse().map(loadPost));

  // remove any nulls (failed files)
  const validPosts = posts.filter(p => p);

  // newest first
  validPosts.sort((a, b) => new Date(b.date) - new Date(a.date));

  for (const post of validPosts) {
    const article = document.createElement("article");
    article.innerHTML = `
      <h2>${post.title}</h2>
      <div class="time">${post.date}</div>
      ${post.html}
    `;
    entriesEl.appendChild(article);
  }
}

loadAll();
